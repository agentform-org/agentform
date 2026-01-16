"""ACP Compiler - Compilation and validation for ACP.

Compiles native ACP schema (.acp) files.
"""

from acp_compiler.acp_ast import MergeError, merge_acp_files
from acp_compiler.acp_normalizer import NormalizationError, normalize_acp

# ACP native schema support
from acp_compiler.acp_parser import (
    ACPParseError,
    discover_acp_files,
    parse_acp,
    parse_acp_directory,
    parse_acp_file,
)
from acp_compiler.acp_resolver import ResolutionError, ResolutionResult, resolve_references
from acp_compiler.acp_validator import ACPValidationError, ACPValidationResult, validate_acp
from acp_compiler.compiler import (
    CompilationError,
    compile_acp,
    compile_acp_directory,
    compile_acp_file,
    compile_file,
    validate_acp_directory,
    validate_acp_file,
    validate_file,
)
from acp_compiler.credentials import (
    CredentialError,
    get_env_var_name,
    is_env_reference,
    resolve_env_var,
)
from acp_compiler.ir_generator import IRGenerationError, generate_ir
from acp_compiler.validator import ValidationError, ValidationResult, validate_spec

__all__ = [
    "ACPParseError",
    "ACPValidationError",
    "ACPValidationResult",
    "CompilationError",
    "CredentialError",
    "IRGenerationError",
    "MergeError",
    "NormalizationError",
    "ResolutionError",
    "ResolutionResult",
    "ValidationError",
    "ValidationResult",
    "compile_acp",
    "compile_acp_directory",
    "compile_acp_file",
    "compile_file",
    "discover_acp_files",
    "generate_ir",
    "get_env_var_name",
    "is_env_reference",
    "merge_acp_files",
    "normalize_acp",
    "parse_acp",
    "parse_acp_directory",
    "parse_acp_file",
    "resolve_env_var",
    "resolve_references",
    "validate_acp",
    "validate_acp_directory",
    "validate_acp_file",
    "validate_file",
    "validate_spec",
]
