"""ACP Compiler - Compilation and validation for ACP.

Supports both YAML (.yaml/.yml) and native ACP schema (.acp) files.
"""

from acp_compiler.compiler import (
    CompilationError,
    compile_acp,
    compile_acp_file,
    compile_file,
    compile_spec,
    compile_spec_file,
    validate_acp_file,
    validate_file,
    validate_spec_file,
)
from acp_compiler.credentials import (
    CredentialError,
    get_env_var_name,
    is_env_reference,
    resolve_env_var,
)
from acp_compiler.ir_generator import IRGenerationError, generate_ir
from acp_compiler.parser import ParseError, parse_yaml, parse_yaml_file
from acp_compiler.validator import ValidationError, ValidationResult, validate_spec

# ACP native schema support
from acp_compiler.acp_parser import ACPParseError, parse_acp, parse_acp_file
from acp_compiler.acp_resolver import ResolutionError, ResolutionResult, resolve_references
from acp_compiler.acp_validator import ACPValidationError, ACPValidationResult, validate_acp
from acp_compiler.acp_normalizer import NormalizationError, normalize_acp

__all__ = [
    # Errors
    "ACPParseError",
    "ACPValidationError",
    "CompilationError",
    "CredentialError",
    "IRGenerationError",
    "NormalizationError",
    "ParseError",
    "ResolutionError",
    "ValidationError",
    # Results
    "ACPValidationResult",
    "ResolutionResult",
    "ValidationResult",
    # YAML functions
    "compile_spec",
    "compile_spec_file",
    "parse_yaml",
    "parse_yaml_file",
    "validate_spec",
    "validate_spec_file",
    # ACP functions
    "compile_acp",
    "compile_acp_file",
    "normalize_acp",
    "parse_acp",
    "parse_acp_file",
    "resolve_references",
    "validate_acp",
    "validate_acp_file",
    # Unified functions (auto-detect format)
    "compile_file",
    "validate_file",
    # IR generation
    "generate_ir",
    # Credentials
    "get_env_var_name",
    "is_env_reference",
    "resolve_env_var",
]
