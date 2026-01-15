"""Main compiler module that ties together parsing, validation, and IR generation."""

from pathlib import Path

from acp_schema.ir import CompiledSpec
from acp_schema.models import SpecRoot

from acp_compiler.ir_generator import generate_ir
from acp_compiler.parser import parse_yaml, parse_yaml_file, ParseError
from acp_compiler.validator import validate_spec, ValidationResult


class CompilationError(Exception):
    """Error during compilation."""

    def __init__(self, message: str, validation_result: ValidationResult | None = None):
        super().__init__(message)
        self.validation_result = validation_result


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

