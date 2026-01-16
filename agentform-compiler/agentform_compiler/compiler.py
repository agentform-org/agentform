"""Main compiler module that ties together parsing, validation, and IR generation."""

from pathlib import Path
from typing import Any

from agentform_compiler.ir_generator import generate_ir
from agentform_compiler.validator import ValidationResult, validate_spec
from agentform_schema.ir import CompiledSpec
from agentform_schema.models import SpecRoot


class CompilationError(Exception):
    """Error during compilation."""

    def __init__(self, message: str, validation_result: ValidationResult | None = None):
        super().__init__(message)
        self.validation_result = validation_result


# File extension detection
Agentform_EXTENSIONS = {".agentform"}


def _is_agentform_file(path: Path) -> bool:
    """Check if a file is an Agentform native schema file."""
    return path.suffix.lower() in Agentform_EXTENSIONS


def parse_agentform_to_spec(
    content: str,
    file_path: str | None = None,
    variables: dict[str, Any] | None = None,
) -> SpecRoot:
    """Parse Agentform content to SpecRoot.

    Args:
        content: Agentform file content as string
        file_path: Optional file path for error messages
        variables: Dictionary of variable values to substitute

    Returns:
        SpecRoot model

    Raises:
        CompilationError: If parsing fails
    """
    from agentform_compiler.agentform_normalizer import NormalizationError, normalize_agentform
    from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform
    from agentform_compiler.agentform_resolver import resolve_references
    from agentform_compiler.agentform_validator import validate_agentform

    # Parse
    try:
        agentform_file = parse_agentform(content, file_path=file_path)
    except AgentformParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Resolve references
    resolution = resolve_references(agentform_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Validate
    validation = validate_agentform(agentform_file, resolution)
    if not validation.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in validation.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}")

    # Normalize to SpecRoot
    try:
        return normalize_agentform(agentform_file, resolution, variables)
    except NormalizationError as e:
        raise CompilationError(f"Normalization error: {e}") from e


def compile_agentform(
    content: str,
    file_path: str | None = None,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile Agentform content to IR.

    Args:
        content: Agentform string content
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
    spec = parse_agentform_to_spec(content, file_path, variables)

    # Validate using existing validator
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def compile_agentform_file(
    path: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile an Agentform file to IR.

    Args:
        path: Path to Agentform file
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

    return compile_agentform(
        content,
        file_path=str(path),
        check_env=check_env,
        resolve_credentials=resolve_credentials,
        variables=variables,
    )


def validate_agentform_file(
    path: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate an Agentform file without full compilation.

    Args:
        path: Path to Agentform file
        check_env: Whether to check env vars exist
        variables: Dictionary of variable values to substitute

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing fails
    """
    from agentform_compiler.agentform_normalizer import NormalizationError, normalize_agentform
    from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform_file
    from agentform_compiler.agentform_resolver import resolve_references
    from agentform_compiler.agentform_validator import validate_agentform

    path = Path(path)

    if not path.exists():
        raise CompilationError(f"File not found: {path}")

    # Parse
    try:
        agentform_file = parse_agentform_file(path)
    except AgentformParseError as e:
        raise CompilationError(f"Parse error: {e}") from e

    # Resolve references
    resolution = resolve_references(agentform_file)
    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Agentform-specific validation
    agentform_validation = validate_agentform(agentform_file, resolution)

    # Convert to ValidationResult format
    result = ValidationResult()
    for agentform_error in agentform_validation.errors:
        result.add_error(agentform_error.path, agentform_error.message)
    for agentform_warning in agentform_validation.warnings:
        result.add_warning(agentform_warning.path, agentform_warning.message)

    if not result.is_valid:
        return result

    # Normalize and run standard validation
    try:
        spec = normalize_agentform(agentform_file, resolution, variables)
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


def compile_agentform_directory(
    directory: str | Path,
    check_env: bool = True,
    resolve_credentials: bool = True,
    variables: dict[str, Any] | None = None,
) -> CompiledSpec:
    """Compile all Agentform files in a directory to IR.

    This function discovers all .agentform files in the directory, parses and merges
    them, then compiles the merged specification. Files are processed in
    alphabetical order for consistent results.

    Args:
        directory: Path to directory containing .agentform files
        check_env: Whether to check env vars exist during validation
        resolve_credentials: Whether to resolve credentials to actual values
        variables: Dictionary of variable values to substitute

    Returns:
        Compiled specification (IR)

    Raises:
        CompilationError: If compilation fails
    """
    from agentform_compiler.agentform_ast import MergeError
    from agentform_compiler.agentform_module_loader import (
        LoadedModule,
        ModuleLoader,
        ModuleLoadError,
    )
    from agentform_compiler.agentform_normalizer import NormalizationError, normalize_agentform
    from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform_directory
    from agentform_compiler.agentform_resolver import add_module_symbols, resolve_references
    from agentform_compiler.agentform_validator import validate_agentform

    directory = Path(directory)

    if not directory.exists():
        raise CompilationError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise CompilationError(f"Path is not a directory: {directory}")

    # Parse and merge all files in directory
    try:
        agentform_file = parse_agentform_directory(directory)
    except AgentformParseError as e:
        raise CompilationError(f"Parse error: {e}") from e
    except MergeError as e:
        raise CompilationError(f"Merge error: {e}") from e

    # Load modules if any are declared
    loaded_modules: dict[str, LoadedModule] = {}
    if agentform_file.modules:
        try:
            loader = ModuleLoader(base_path=directory)
            loaded_modules = loader.load_modules(agentform_file.modules)
        except ModuleLoadError as e:
            raise CompilationError(f"Module load error: {e}") from e

    # Resolve references (builds initial symbol table)
    resolution = resolve_references(agentform_file)

    # Add module symbols to the resolution result
    for module_name, loaded_module in loaded_modules.items():
        add_module_symbols(resolution, module_name, loaded_module.agentform_file)

    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Validate
    validation = validate_agentform(agentform_file, resolution)
    if not validation.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in validation.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}")

    # Normalize to SpecRoot (with module support)
    try:
        spec = normalize_agentform(agentform_file, resolution, variables, loaded_modules)
    except NormalizationError as e:
        raise CompilationError(f"Normalization error: {e}") from e

    # Validate using existing validator
    result = validate_spec(spec, check_env=check_env)
    if not result.is_valid:
        errors_str = "\n".join(f"  - {e.path}: {e.message}" for e in result.errors)
        raise CompilationError(f"Validation failed:\n{errors_str}", result)

    # Generate IR
    return generate_ir(spec, resolve_credentials=resolve_credentials)


def validate_agentform_directory(
    directory: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate all Agentform files in a directory without full compilation.

    This function discovers all .agentform files in the directory, parses and merges
    them, then validates the merged specification.

    Args:
        directory: Path to directory containing .agentform files
        check_env: Whether to check env vars exist
        variables: Dictionary of variable values to substitute

    Returns:
        ValidationResult with errors and warnings

    Raises:
        CompilationError: If parsing or merging fails
    """
    from agentform_compiler.agentform_ast import MergeError
    from agentform_compiler.agentform_module_loader import (
        LoadedModule,
        ModuleLoader,
        ModuleLoadError,
    )
    from agentform_compiler.agentform_normalizer import NormalizationError, normalize_agentform
    from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform_directory
    from agentform_compiler.agentform_resolver import add_module_symbols, resolve_references
    from agentform_compiler.agentform_validator import validate_agentform

    directory = Path(directory)

    if not directory.exists():
        raise CompilationError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise CompilationError(f"Path is not a directory: {directory}")

    # Parse and merge all files in directory
    try:
        agentform_file = parse_agentform_directory(directory)
    except AgentformParseError as e:
        raise CompilationError(f"Parse error: {e}") from e
    except MergeError as e:
        raise CompilationError(f"Merge error: {e}") from e

    # Load modules if any are declared
    loaded_modules: dict[str, LoadedModule] = {}
    if agentform_file.modules:
        try:
            loader = ModuleLoader(base_path=directory)
            loaded_modules = loader.load_modules(agentform_file.modules)
        except ModuleLoadError as e:
            raise CompilationError(f"Module load error: {e}") from e

    # Resolve references
    resolution = resolve_references(agentform_file)

    # Add module symbols to the resolution result
    for module_name, loaded_module in loaded_modules.items():
        add_module_symbols(resolution, module_name, loaded_module.agentform_file)

    if not resolution.is_valid:
        errors_str = "\n".join(f"  - {e}" for e in resolution.errors)
        raise CompilationError(f"Reference resolution failed:\n{errors_str}")

    # Agentform-specific validation
    agentform_validation = validate_agentform(agentform_file, resolution)

    # Convert to ValidationResult format
    result = ValidationResult()
    for agentform_error in agentform_validation.errors:
        result.add_error(agentform_error.path, agentform_error.message)
    for agentform_warning in agentform_validation.warnings:
        result.add_warning(agentform_warning.path, agentform_warning.message)

    if not result.is_valid:
        return result

    # Normalize and run standard validation (with module support)
    try:
        spec = normalize_agentform(agentform_file, resolution, variables, loaded_modules)
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
    """Compile an Agentform file or directory to IR.

    If path is a directory, all .agentform files in it are discovered, parsed,
    merged, and compiled together (Terraform-style multi-file support).

    Args:
        path: Path to .agentform spec file or directory containing .agentform files
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
        return compile_agentform_directory(path, check_env, resolve_credentials, variables)

    # Handle single file compilation
    if not _is_agentform_file(path):
        raise CompilationError(
            f"Expected .agentform file, got: {path.suffix}. Only .agentform files are supported."
        )

    return compile_agentform_file(path, check_env, resolve_credentials, variables)


def validate_file(
    path: str | Path,
    check_env: bool = True,
    variables: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate an Agentform file or directory.

    If path is a directory, all .agentform files in it are discovered, parsed,
    merged, and validated together (Terraform-style multi-file support).

    Args:
        path: Path to .agentform spec file or directory containing .agentform files
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
        return validate_agentform_directory(path, check_env, variables)

    # Handle single file validation
    if not _is_agentform_file(path):
        raise CompilationError(
            f"Expected .agentform file, got: {path.suffix}. Only .agentform files are supported."
        )

    return validate_agentform_file(path, check_env, variables)
