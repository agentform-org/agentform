"""Module resolver for ACP module system.

Handles resolving module sources from Git URLs or local paths,
with caching support for Git-based modules.

Supports Terraform-style source syntax:
- Local paths: "./modules/my-module"
- Git URLs: "github.com/org/repo"
- Git URLs with subdirectory: "github.com/org/repo//path/to/module"

Modules are cached locally in .acp/modules/ within the project directory.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import NamedTuple


class ModuleResolutionError(Exception):
    """Error during module resolution."""

    pass


class ResolvedModule(NamedTuple):
    """Result of module resolution."""

    path: Path  # Resolved local path to module directory
    source: str  # Original source string
    version: str | None  # Version/ref if specified
    is_local: bool  # Whether this is a local path or Git module


class ParsedGitSource(NamedTuple):
    """Parsed Git source URL."""

    repo_url: str  # Git repository URL (for cloning)
    subdir: str | None  # Subdirectory within the repo (after //)


# Git URL patterns
_GIT_URL_PATTERNS = [
    # github.com/org/repo
    r"^github\.com/[^/]+/[^/]+",
    # gitlab.com/org/repo
    r"^gitlab\.com/[^/]+/[^/]+",
    # bitbucket.org/org/repo
    r"^bitbucket\.org/[^/]+/[^/]+",
    # https://github.com/...
    r"^https?://github\.com/[^/]+/[^/]+",
    r"^https?://gitlab\.com/[^/]+/[^/]+",
    r"^https?://bitbucket\.org/[^/]+/[^/]+",
    # git@github.com:org/repo.git
    r"^git@[^:]+:.+\.git$",
    # Generic https Git URL
    r"^https?://.+\.git$",
]


def is_git_url(source: str) -> bool:
    """Check if a source string looks like a Git URL.

    Args:
        source: Module source string

    Returns:
        True if this appears to be a Git URL
    """
    return any(re.match(pattern, source) for pattern in _GIT_URL_PATTERNS)


def _parse_git_source(source: str) -> ParsedGitSource:
    """Parse a Git source URL, extracting repo URL and optional subdirectory.

    Supports Terraform-style syntax with // for subdirectories:
    - "github.com/org/repo" -> repo only
    - "github.com/org/repo//subdir" -> repo with subdirectory
    - "github.com/org/repo//path/to/module" -> repo with nested subdirectory

    Args:
        source: Module source string

    Returns:
        ParsedGitSource with repo_url and optional subdir
    """
    # Check for // subdirectory separator
    if "//" in source:
        parts = source.split("//", 1)
        repo_url = parts[0].rstrip("/")
        subdir = parts[1].lstrip("/") if len(parts) > 1 else None
        return ParsedGitSource(repo_url=repo_url, subdir=subdir)

    return ParsedGitSource(repo_url=source, subdir=None)


def _normalize_git_url(source: str) -> str:
    """Normalize a Git URL to HTTPS format for cloning.

    Args:
        source: Original source string (e.g., "github.com/org/repo")

    Returns:
        Full HTTPS URL for cloning
    """
    # Already a full URL
    if source.startswith("https://") or source.startswith("http://"):
        return source

    # SSH format: git@github.com:org/repo.git
    if source.startswith("git@"):
        # Convert git@github.com:org/repo.git to https://github.com/org/repo.git
        match = re.match(r"^git@([^:]+):(.+)$", source)
        if match:
            host, path = match.groups()
            return f"https://{host}/{path}"

    # Short format: github.com/org/repo
    if source.startswith("github.com/"):
        return f"https://{source}"
    if source.startswith("gitlab.com/"):
        return f"https://{source}"
    if source.startswith("bitbucket.org/"):
        return f"https://{source}"

    # Assume it's already a valid URL
    return source


def _get_cache_key(source: str, version: str | None) -> str:
    """Generate a cache key for a module source.

    Args:
        source: Module source URL
        version: Version string (tag, branch, commit)

    Returns:
        Cache key string
    """
    # Normalize the source URL
    normalized = _normalize_git_url(source)

    # Create a hash of the source + version for uniqueness
    key_input = f"{normalized}@{version or 'HEAD'}"
    hash_suffix = hashlib.sha256(key_input.encode()).hexdigest()[:12]

    # Create a readable name from the source
    # github.com/org/repo -> org_repo
    name_part = re.sub(r"^https?://", "", normalized)
    name_part = re.sub(r"\.git$", "", name_part)
    name_part = name_part.replace("/", "_").replace(".", "_")

    return f"{name_part}_{hash_suffix}"


def get_cache_dir(base_path: Path | None = None) -> Path:
    """Get the module cache directory.

    Uses .acp/modules/ within the project directory (like Terraform).
    Falls back to ACP_MODULE_CACHE_DIR env var if set.

    Args:
        base_path: Project base path. If None, uses current directory.

    Returns:
        Path to cache directory
    """
    if cache_dir := os.environ.get("ACP_MODULE_CACHE_DIR"):
        return Path(cache_dir)

    # Use local .acp/modules/ directory within the project
    project_path = base_path or Path.cwd()
    return project_path / ".acp" / "modules"


class ModuleResolver:
    """Resolves module sources to local paths.

    Handles both local paths and Git URLs, with caching for Git modules.
    Modules are cached in .acp/modules/ within the project directory.

    Git modules must be downloaded first using 'acp init'.
    Use download_module() to download modules (called by init command).
    """

    def __init__(
        self,
        base_path: Path | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize the module resolver.

        Args:
            base_path: Base path for resolving relative local paths.
                      Defaults to current working directory.
            cache_dir: Directory to cache Git modules.
                      Defaults to .acp/modules/ in the project directory
        """
        self.base_path = base_path or Path.cwd()
        self.cache_dir = cache_dir or get_cache_dir(self.base_path)
        self._resolved_cache: dict[tuple[str, str | None], ResolvedModule] = {}

    def resolve(self, source: str, version: str | None = None) -> ResolvedModule:
        """Resolve a module source to a local path.

        For Git modules, this will error if the module hasn't been downloaded.
        Use 'acp init' to download modules first.

        Args:
            source: Module source (Git URL or local path)
            version: Version/ref for Git modules (tag, branch, commit)

        Returns:
            ResolvedModule with the local path

        Raises:
            ModuleResolutionError: If resolution fails or module not downloaded
        """
        # Check cache first
        cache_key = (source, version)
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        if is_git_url(source):
            result = self._resolve_git_module(source, version)
        else:
            result = self._resolve_local_module(source)

        self._resolved_cache[cache_key] = result
        return result

    def download_module(self, source: str, version: str | None = None) -> ResolvedModule:
        """Download a Git module (used by 'acp init').

        This method always clones/updates Git modules, regardless of auto_download setting.
        For local modules, this just validates they exist.

        Args:
            source: Module source (Git URL or local path)
            version: Version/ref for Git modules (tag, branch, commit)

        Returns:
            ResolvedModule with the local path

        Raises:
            ModuleResolutionError: If download fails
        """
        if not is_git_url(source):
            # Local module - just resolve it
            return self._resolve_local_module(source)

        # Git module - force download
        # Parse source to extract repo URL and subdirectory
        parsed = _parse_git_source(source)
        repo_url = parsed.repo_url
        subdir = parsed.subdir

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get cache key and target path
        cache_key = _get_cache_key(repo_url, version)
        repo_path = self.cache_dir / cache_key

        # Normalize URL for cloning
        git_url = _normalize_git_url(repo_url)

        # Always clone or update
        if repo_path.exists():
            if not (repo_path / ".git").exists():
                # Not a valid git repo, remove and re-clone
                shutil.rmtree(repo_path)
                self._clone_module(git_url, repo_path, version)
            elif version:
                # Update to specific version
                self._checkout_version(repo_path, version)
            # else: already exists and no specific version, keep as-is
        else:
            self._clone_module(git_url, repo_path, version)

        # Determine module path (repo root or subdirectory)
        if subdir:
            module_path = repo_path / subdir
            if not module_path.exists():
                raise ModuleResolutionError(
                    f"Subdirectory '{subdir}' not found in repository: {repo_url}"
                )
            if not module_path.is_dir():
                raise ModuleResolutionError(f"Subdirectory path is not a directory: {subdir}")
        else:
            module_path = repo_path

        # Verify module has .acp files
        acp_files = list(module_path.glob("*.acp"))
        if not acp_files:
            raise ModuleResolutionError(f"No .acp files found in module: {source}")

        return ResolvedModule(
            path=module_path,
            source=source,
            version=version,
            is_local=False,
        )

    def _resolve_local_module(self, source: str) -> ResolvedModule:
        """Resolve a local module path.

        Args:
            source: Local path (relative or absolute)

        Returns:
            ResolvedModule with resolved path

        Raises:
            ModuleResolutionError: If path doesn't exist
        """
        path = Path(source)

        # Resolve relative paths against base_path
        if not path.is_absolute():
            path = self.base_path / path

        # Resolve symlinks and normalize
        path = path.resolve()

        if not path.exists():
            raise ModuleResolutionError(f"Module path does not exist: {source}")

        if not path.is_dir():
            raise ModuleResolutionError(f"Module path is not a directory: {source}")

        # Check for .acp files
        acp_files = list(path.glob("*.acp"))
        if not acp_files:
            raise ModuleResolutionError(f"No .acp files found in module directory: {source}")

        return ResolvedModule(
            path=path,
            source=source,
            version=None,
            is_local=True,
        )

    def _resolve_git_module(self, source: str, version: str | None) -> ResolvedModule:
        """Resolve a Git module by cloning/updating it.

        Supports Terraform-style // syntax for subdirectories:
        - "github.com/org/repo" -> clones repo, uses root
        - "github.com/org/repo//subdir" -> clones repo, uses subdir

        Args:
            source: Git URL (optionally with //subdir)
            version: Git ref (tag, branch, commit)

        Returns:
            ResolvedModule with path to cloned module

        Raises:
            ModuleResolutionError: If cloning fails
        """
        # Parse source to extract repo URL and subdirectory
        parsed = _parse_git_source(source)
        repo_url = parsed.repo_url
        subdir = parsed.subdir

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get cache key and target path (based on repo URL only)
        cache_key = _get_cache_key(repo_url, version)
        repo_path = self.cache_dir / cache_key

        # Check if module is cached (must run 'acp init' first)
        if not repo_path.exists():
            raise ModuleResolutionError(
                f"Module '{source}' not found locally. Run 'acp init' to download external modules."
            )

        # Verify it's a valid git repo
        if not (repo_path / ".git").exists():
            raise ModuleResolutionError(
                f"Module '{source}' cache is corrupted. Run 'acp init' to re-download modules."
            )

        # Checkout specific version if requested
        if version:
            self._checkout_version(repo_path, version)

        # Determine the module path (repo root or subdirectory)
        if subdir:
            module_path = repo_path / subdir
            if not module_path.exists():
                raise ModuleResolutionError(
                    f"Subdirectory '{subdir}' not found in repository: {repo_url}"
                )
            if not module_path.is_dir():
                raise ModuleResolutionError(f"Subdirectory path is not a directory: {subdir}")
        else:
            module_path = repo_path

        # Verify module has .acp files
        acp_files = list(module_path.glob("*.acp"))
        if not acp_files:
            raise ModuleResolutionError(f"No .acp files found in module: {source}")

        return ResolvedModule(
            path=module_path,
            source=source,
            version=version,
            is_local=False,
        )

    def _clone_module(self, url: str, target_path: Path, version: str | None) -> None:
        """Clone a Git repository.

        Args:
            url: Git URL to clone
            target_path: Target directory
            version: Git ref to checkout

        Raises:
            ModuleResolutionError: If cloning fails
        """
        try:
            # Clone with depth=1 for efficiency (unless we need history)
            cmd = ["git", "clone", "--depth", "1"]

            if version:
                # Try to clone specific branch/tag
                cmd.extend(["--branch", version])

            cmd.extend([url, str(target_path)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if result.returncode != 0:
                # If branch clone failed, try full clone + checkout
                if version and "not found" in result.stderr.lower():
                    self._clone_and_checkout(url, target_path, version)
                else:
                    raise ModuleResolutionError(f"Failed to clone module {url}: {result.stderr}")

        except subprocess.TimeoutExpired as e:
            raise ModuleResolutionError(f"Timeout while cloning module {url}") from e
        except FileNotFoundError as e:
            raise ModuleResolutionError("Git is not installed or not in PATH") from e

    def _clone_and_checkout(self, url: str, target_path: Path, version: str) -> None:
        """Clone a repository and checkout a specific ref.

        Used when the ref might be a commit hash or non-branch ref.

        Args:
            url: Git URL to clone
            target_path: Target directory
            version: Git ref to checkout

        Raises:
            ModuleResolutionError: If operation fails
        """
        try:
            # Full clone without depth limit
            result = subprocess.run(
                ["git", "clone", url, str(target_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for full clone
            )

            if result.returncode != 0:
                raise ModuleResolutionError(f"Failed to clone module {url}: {result.stderr}")

            # Checkout the specific version
            self._checkout_version(target_path, version)

        except subprocess.TimeoutExpired as e:
            raise ModuleResolutionError(f"Timeout while cloning module {url}") from e

    def _checkout_version(self, repo_path: Path, version: str) -> None:
        """Checkout a specific version in a Git repository.

        Args:
            repo_path: Path to Git repository
            version: Git ref to checkout

        Raises:
            ModuleResolutionError: If checkout fails
        """
        try:
            # Fetch to ensure we have latest refs
            fetch_result = subprocess.run(
                ["git", "fetch", "--all", "--tags"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if fetch_result.returncode != 0:
                raise ModuleResolutionError(
                    f"Failed to fetch refs for module: {fetch_result.stderr}"
                )

            # Checkout the version
            result = subprocess.run(
                ["git", "checkout", version],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise ModuleResolutionError(
                    f"Failed to checkout version {version}: {result.stderr}"
                )

        except subprocess.TimeoutExpired as e:
            raise ModuleResolutionError(f"Timeout while checking out version {version}") from e

    def clear_cache(self) -> None:
        """Clear the in-memory resolution cache."""
        self._resolved_cache.clear()


def resolve_module_source(
    source: str,
    version: str | None = None,
    base_path: Path | None = None,
    cache_dir: Path | None = None,
) -> ResolvedModule:
    """Convenience function to resolve a module source.

    Args:
        source: Module source (Git URL or local path)
        version: Version/ref for Git modules
        base_path: Base path for relative local paths
        cache_dir: Cache directory for Git modules

    Returns:
        ResolvedModule with the local path

    Raises:
        ModuleResolutionError: If resolution fails
    """
    resolver = ModuleResolver(base_path=base_path, cache_dir=cache_dir)
    return resolver.resolve(source, version)
