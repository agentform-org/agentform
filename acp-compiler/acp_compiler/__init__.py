"""ACP Compiler - YAML compilation and validation for ACP."""

from acp_compiler.parser import parse_yaml, parse_yaml_file, ParseError
from acp_compiler.validator import validate_spec, ValidationResult, ValidationError
from acp_compiler.compiler import (
    compile_spec,
    compile_spec_file,
    validate_spec_file,
    CompilationError,
)
from acp_compiler.ir_generator import generate_ir, IRGenerationError
from acp_compiler.credentials import (
    is_env_reference,
    get_env_var_name,
    resolve_env_var,
    CredentialError,
)

__all__ = [
    "parse_yaml",
    "parse_yaml_file",
    "ParseError",
    "validate_spec",
    "ValidationResult",
    "ValidationError",
    "compile_spec",
    "compile_spec_file",
    "validate_spec_file",
    "CompilationError",
    "generate_ir",
    "IRGenerationError",
    "is_env_reference",
    "get_env_var_name",
    "resolve_env_var",
    "CredentialError",
]
