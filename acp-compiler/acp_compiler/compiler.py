"""Main compiler module that ties together parsing, validation, and IR generation."""

from pathlib import Path
from typing import Any

from acp_compiler.ir_generator import generate_ir
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


def _is_acp_file(path: Path) -> bool:
    """Check if a file is an ACP native schema file."""
    return path.suffix.lower() in ACP_EXTENSIONS


def parse_acp_to_spec(
    content: str,
    file_path: str | None = None,
    variables: dict[str, Any] | None = None,
) -> SpecRoot:
    """Parse ACP content to SpecRoot.

    Args:
        content: ACP file content as string
        file_path: Optional file path for error messages
        variables: Dictionary of variable values to substitute

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
    validation = validate_acp(acp_file, resolution)
    if not validation.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in validation.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}")

    # Normalize to SpecRoot
    try:
        return normalize_acp(acp_file, resolution, variables)
    except NormalizationError as e:
        raise CompilationError(f"Normalization error: {e}") from e


def compile_acp(
    content: str,
    file_path: str | None = None,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile ACP content to IR.

    Args:
        content: ACP string content
        file_path: Optional file path for error messages
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values
        variables: Dictionary of variable values to substitute

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    # Parse and normalize to SpecRoot
    spec = parse_acp_to_spec(content, file_path, variables)

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
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile an ACP file to IR.

    Args:
        path: Path to ACP file
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values
        variables: Dictionary of variable values to substitute

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """

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
        variables=variables,
    )


def validate_acp_file(
    path: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate an ACP file without full compilation.

    Args:
        path: Path to ACP file
        check_env: Whether to check env vars exist
        variables: Dictionary of variable values to substitute

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
    acp_validation = validate_acp(acp_file, resolution)

    # Convert to ValidationResult format
    result = ValidationResult()
    for acp_error in acp_validation.errors:
        result.add_error(acp_error.path, acp_error.message)
    for acp_warning in acp_validation.warnings:
        result.add_warning(acp_warning.path, acp_warning.message)

    if not result.is_valid:
        return result

    # Normalize and run standard validation
    try:
        spec = normalize_acp(acp_file, resolution, variables)
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
# Directory Compilation (Multi-File Support)
# ============================================================================


def compile_acp_directory(
    directory: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile all ACP files in a directory to IR.

    This function discovers all .acp files in the directory, parses and merges
    them, then compiles the merged specification. Files are processed in
    alphabetical order for consistent results.

    Args:
        directory: Path to directory containing .acp files
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values
        variables: Dictionary of variable values to substitute

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    from acp_compiler.acp_ast import MergeError
    from acp_compiler.acp_normalizer import NormalizationError, normalize_acp
    from acp_compiler.acp_parser import ACPParseError, parse_acp_directory
    from acp_compiler.acp_resolver import resolve_references
    from acp_compiler.acp_validator import validate_acp

    directory = Path(directory)

    if not directory.exists():
        raise CompilationError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise CompilationError(f"Path is not a directory: {directory}")

    # Parse and merge all files in directory
    try:
        acp_file = parse_acp_directory(directory)
    except ACPParseError as e:
        raise CompilationError(f"Parse error: {e}") from e
    except MergeError as e:
        raise CompilationError(f"Merge error: {e}") from e

    # Resolve references
    resolution = resolve_references(acp_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Validate
    validation = validate_acp(acp_file, resolution)
    if not validation.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in validation.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}")

    # Normalize to SpecRoot
    try:
        spec = normalize_acp(acp_file, resolution, variables)
    except NormalizationError as e:
        raise CompilationError(f"Normalization error: {e}") from e

    # Validate using existing validator
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def validate_acp_directory(
    directory: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate all ACP files in a directory without full compilation.

    This function discovers all .acp files in the directory, parses and merges
    them, then validates the merged specification.

    Args:
        directory: Path to directory containing .acp files
        check_env: Whether to check env vars exist
        variables: Dictionary of variable values to substitute

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing or merging fails
    """
    from acp_compiler.acp_ast import MergeError
    from acp_compiler.acp_normalizer import NormalizationError, normalize_acp
    from acp_compiler.acp_parser import ACPParseError, parse_acp_directory
    from acp_compiler.acp_resolver import resolve_references
    from acp_compiler.acp_validator import validate_acp

    directory = Path(directory)

    if not directory.exists():
        raise CompilationError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise CompilationError(f"Path is not a directory: {directory}")

    # Parse and merge all files in directory
    try:
        acp_file = parse_acp_directory(directory)
    except ACPParseError as e:
        raise CompilationError(f"Parse error: {e}") from e
    except MergeError as e:
        raise CompilationError(f"Merge error: {e}") from e

    # Resolve references
    resolution = resolve_references(acp_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # ACP-specific validation
    acp_validation = validate_acp(acp_file, resolution)

    # Convert to ValidationResult format
    result = ValidationResult()
    for acp_error in acp_validation.errors:
        result.add_error(acp_error.path, acp_error.message)
    for acp_warning in acp_validation.warnings:
        result.add_warning(acp_warning.path, acp_warning.message)

    if not result.is_valid:
        return result

    # Normalize and run standard validation
    try:
        spec = normalize_acp(acp_file, resolution, variables)
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
# Unified Interface
# ============================================================================


def compile_file(
    path: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile an ACP file or directory to IR.

    If path is a directory, all .acp files in it are discovered, parsed,
    merged, and compiled together (Terraform-style multi-file support).

    Args:
        path: Path to .acp spec file or directory containing .acp files
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values
        variables: Dictionary of variable values to substitute

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    path = Path(path)

    # Handle directory compilation
    if path.is_dir():
        return compile_acp_directory(path, check_env, resolve_credentials, variables)

    # Handle single file compilation
    if not _is_acp_file(path):
        raise CompilationError(
            f"Expected .acp file, got: {path.suffix}. Only .acp files are supported."
        )

    return compile_acp_file(path, check_env, resolve_credentials, variables)


def validate_file(
    path: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate an ACP file or directory.

    If path is a directory, all .acp files in it are discovered, parsed,
    merged, and validated together (Terraform-style multi-file support).

    Args:
        path: Path to .acp spec file or directory containing .acp files
        check_env: Whether to check env vars exist
        variables: Dictionary of variable values to substitute

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing fails
    """
    path = Path(path)

    # Handle directory validation
    if path.is_dir():
        return validate_acp_directory(path, check_env, variables)

    # Handle single file validation
    if not _is_acp_file(path):
        raise CompilationError(
            f"Expected .acp file, got: {path.suffix}. Only .acp files are supported."
        )

    return validate_acp_file(path, check_env, variables)
