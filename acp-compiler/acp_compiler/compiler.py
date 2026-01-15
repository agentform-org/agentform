"""Main compiler module that ties together parsing, validation, and IR generation."""

from pathlib import Path

from acp_compiler.ir_generator import generate_ir
from acp_compiler.parser import ParseError, parse_yaml, parse_yaml_file
from acp_compiler.validator import ValidationResult, validate_spec
from acp_schema.ir import CompiledSpec
from acp_schema.models import SpecRoot


class CompilationError(Exception):
    """Error during compilation."""

    def __init__(self, message: str, validation_result: ValidationResult | None = None):
        super().__init__(message)
        self.validation_result = validation_result


# File extension detection
ACP_EXTENSIONS = {".acp"}
YAML_EXTENSIONS = {".yaml", ".yml"}


def _is_acp_file(path: Path) -> bool:
    """Check if a file is an ACP native schema file."""
    return path.suffix.lower() in ACP_EXTENSIONS


def _is_yaml_file(path: Path) -> bool:
    """Check if a file is a YAML file."""
    return path.suffix.lower() in YAML_EXTENSIONS


def compile_spec(
    content: str,
    check_env: bool = True,
    resolve_credentials: bool = True,
) -> CompiledSpec:
    """Compile YAML content to IR.

    Args:
        content: YAML string content
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    # Parse
    try:
        spec = parse_yaml(content)
    except ParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Validate
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def compile_spec_file(
    path: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
) -> CompiledSpec:
    """Compile a YAML file to IR.

    Args:
        path: Path to YAML file
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    # Parse
    try:
        spec = parse_yaml_file(path)
    except ParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Validate
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def validate_spec_file(path: str | Path, check_env: bool = True) -> ValidationResult:
    """Validate a YAML file without full compilation.

    This is faster than compile_spec_file as it doesn't generate IR
    or connect to MCP servers.

    Args:
        path: Path to YAML file
        check_env: Whether to check env vars exist

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing fails
    """
    try:
        spec = parse_yaml_file(path)
    except ParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    return validate_spec(spec, check_env=check_env)


# ============================================================================
# ACP Native Schema Support
# ============================================================================


def parse_acp_to_spec(content: str, file_path: str | None = None) -> SpecRoot:
    """Parse ACP content to SpecRoot.

    Args:
        content: ACP file content as string
        file_path: Optional file path for error messages

    Returns:
        SpecRoot model

    Raises:
        CompilationError: If parsing fails
    """
    from acp_compiler.acp_normalizer import NormalizationError, normalize_acp
    from acp_compiler.acp_parser import ACPParseError, parse_acp
    from acp_compiler.acp_resolver import resolve_references
    from acp_compiler.acp_validator import validate_acp

    # Parse
    try:
        acp_file = parse_acp(content, file_path=file_path)
    except ACPParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Resolve references
    resolution = resolve_references(acp_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Validate
    validation = validate_acp(acp_file, resolution, check_env=False)
    if not validation.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in validation.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}")

    # Normalize to SpecRoot
    try:
        return normalize_acp(acp_file, resolution)
    except NormalizationError as e:
        raise CompilationError(f"Normalization error: {e}") from e


def compile_acp(
    content: str,
    file_path: str | None = None,
    check_env: bool = True,
    resolve_credentials: bool = True,
) -> CompiledSpec:
    """Compile ACP content to IR.

    Args:
        content: ACP string content
        file_path: Optional file path for error messages
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    # Parse and normalize to SpecRoot
    spec = parse_acp_to_spec(content, file_path)

    # Validate using existing validator
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def compile_acp_file(
    path: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
) -> CompiledSpec:
    """Compile an ACP file to IR.

    Args:
        path: Path to ACP file
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    from acp_compiler.acp_parser import ACPParseError

    path = Path(path)

    if not path.exists():
        raise CompilationError(f"File not found: {path}")

    try:
        content = path.read_text()
    except OSError as e:
        raise CompilationError(f"Failed to read file: {e}") from e

    return compile_acp(
        content,
        file_path=str(path),
        check_env=check_env,
        resolve_credentials=resolve_credentials,
    )


def validate_acp_file(path: str | Path, check_env: bool = True) -> ValidationResult:
    """Validate an ACP file without full compilation.

    Args:
        path: Path to ACP file
        check_env: Whether to check env vars exist

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing fails
    """
    from acp_compiler.acp_normalizer import NormalizationError, normalize_acp
    from acp_compiler.acp_parser import ACPParseError, parse_acp_file
    from acp_compiler.acp_resolver import resolve_references
    from acp_compiler.acp_validator import validate_acp

    path = Path(path)

    if not path.exists():
        raise CompilationError(f"File not found: {path}")

    # Parse
    try:
        acp_file = parse_acp_file(path)
    except ACPParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Resolve references
    resolution = resolve_references(acp_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # ACP-specific validation
    acp_validation = validate_acp(acp_file, resolution, check_env=check_env)

    # Convert to ValidationResult format
    result = ValidationResult()
    for error in acp_validation.errors:
        result.add_error(error.path, error.message)
    for warning in acp_validation.warnings:
        result.add_warning(warning.path, warning.message)

    if not result.is_valid:
        return result

    # Normalize and run standard validation
    try:
        spec = normalize_acp(acp_file, resolution)
    except NormalizationError as e:
        result.add_error("normalization", str(e))
        return result

    # Run standard spec validation
    spec_result = validate_spec(spec, check_env=check_env)
    for error in spec_result.errors:
        result.add_error(error.path, error.message)
    for warning in spec_result.warnings:
        result.add_warning(warning.path, warning.message)

    return result


# ============================================================================
# Unified Interface (Auto-detect file format)
# ============================================================================


def compile_file(
    path: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
) -> CompiledSpec:
    """Compile a spec file to IR, auto-detecting format by extension.

    Supports both .acp (native schema) and .yaml/.yml (YAML) files.

    Args:
        path: Path to spec file
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    path = Path(path)

    if _is_acp_file(path):
        return compile_acp_file(path, check_env, resolve_credentials)
    elif _is_yaml_file(path):
        return compile_spec_file(path, check_env, resolve_credentials)
    else:
        raise CompilationError(
            f"Unknown file extension: {path.suffix}. "
            f"Supported: {', '.join(ACP_EXTENSIONS | YAML_EXTENSIONS)}"
        )


def validate_file(path: str | Path, check_env: bool = True) -> ValidationResult:
    """Validate a spec file, auto-detecting format by extension.

    Supports both .acp (native schema) and .yaml/.yml (YAML) files.

    Args:
        path: Path to spec file
        check_env: Whether to check env vars exist

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing fails
    """
    path = Path(path)

    if _is_acp_file(path):
        return validate_acp_file(path, check_env)
    elif _is_yaml_file(path):
        return validate_spec_file(path, check_env)
    else:
        raise CompilationError(
            f"Unknown file extension: {path.suffix}. "
            f"Supported: {', '.join(ACP_EXTENSIONS | YAML_EXTENSIONS)}"
        )
