"""Tests for Agentform module resolver."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler.agentform_module_resolver import (
    ModuleResolutionError,
    ModuleResolver,
    _get_cache_key,
    _normalize_git_url,
    is_git_url,
)


class TestIsGitUrl:
    """Tests for is_git_url function."""

    def test_github_short_url(self) -> None:
        assert is_git_url("github.com/org/repo") is True

    def test_github_https_url(self) -> None:
        assert is_git_url("https://github.com/org/repo") is True

    def test_gitlab_url(self) -> None:
        assert is_git_url("gitlab.com/org/repo") is True

    def test_bitbucket_url(self) -> None:
        assert is_git_url("bitbucket.org/org/repo") is True

    def test_ssh_url(self) -> None:
        assert is_git_url("git@github.com:org/repo.git") is True

    def test_local_relative_path(self) -> None:
        assert is_git_url("./local/path") is False

    def test_local_absolute_path(self) -> None:
        assert is_git_url("/absolute/path") is False

    def test_bare_directory_name(self) -> None:
        assert is_git_url("my-module") is False


class TestNormalizeGitUrl:
    """Tests for _normalize_git_url function."""

    def test_github_short_to_https(self) -> None:
        result = _normalize_git_url("github.com/org/repo")
        assert result == "https://github.com/org/repo"

    def test_ssh_to_https(self) -> None:
        result = _normalize_git_url("git@github.com:org/repo.git")
        assert result == "https://github.com/org/repo.git"

    def test_https_passthrough(self) -> None:
        url = "https://github.com/org/repo"
        result = _normalize_git_url(url)
        assert result == url

    def test_gitlab_short_to_https(self) -> None:
        result = _normalize_git_url("gitlab.com/org/repo")
        assert result == "https://gitlab.com/org/repo"


class TestCacheKey:
    """Tests for _get_cache_key function."""

    def test_includes_version_in_key(self) -> None:
        key1 = _get_cache_key("github.com/org/repo", "v1.0")
        key2 = _get_cache_key("github.com/org/repo", "v2.0")
        assert key1 != key2

    def test_none_version_uses_head(self) -> None:
        key = _get_cache_key("github.com/org/repo", None)
        assert "HEAD" not in key  # Hash includes HEAD but readable part doesn't
        assert "github_com_org_repo" in key

    def test_readable_prefix(self) -> None:
        key = _get_cache_key("github.com/agentform-team/llm-providers", "v1.0")
        # The key will contain the org/repo name (possibly with hyphens still)
        assert "github_com" in key
        assert "agentform" in key


class TestModuleResolverLocal:
    """Tests for local module resolution."""

    def test_resolves_existing_directory(self) -> None:
        # Use the fixture module
        fixtures_dir = Path(__file__).parent / "fixtures" / "modules" / "simple-module"
        if not fixtures_dir.exists():
            pytest.skip("Fixture module not found")

        resolver = ModuleResolver(base_path=fixtures_dir.parent)
        result = resolver.resolve("simple-module")

        assert result.path == fixtures_dir
        assert result.source == "simple-module"
        assert result.version is None
        assert result.is_local is True

    def test_resolves_absolute_path(self) -> None:
        fixtures_dir = Path(__file__).parent / "fixtures" / "modules" / "simple-module"
        if not fixtures_dir.exists():
            pytest.skip("Fixture module not found")

        resolver = ModuleResolver()
        result = resolver.resolve(str(fixtures_dir))

        assert result.path == fixtures_dir.resolve()
        assert result.is_local is True

    def test_raises_for_nonexistent_path(self) -> None:
        resolver = ModuleResolver()

        with pytest.raises(ModuleResolutionError) as exc_info:
            resolver.resolve("/nonexistent/path/to/module")

        assert "does not exist" in str(exc_info.value)

    def test_raises_for_empty_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = ModuleResolver()

            with pytest.raises(ModuleResolutionError) as exc_info:
                resolver.resolve(tmpdir)

            assert "No .af files found" in str(exc_info.value)

    def test_caches_resolution_results(self) -> None:
        fixtures_dir = Path(__file__).parent / "fixtures" / "modules" / "simple-module"
        if not fixtures_dir.exists():
            pytest.skip("Fixture module not found")

        resolver = ModuleResolver(base_path=fixtures_dir.parent)
        result1 = resolver.resolve("simple-module")
        result2 = resolver.resolve("simple-module")

        # Should return the same cached result
        assert result1 is result2
