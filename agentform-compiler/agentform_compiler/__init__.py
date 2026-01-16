"""Agentform Compiler - Compilation and validation for Agentform.

Compiles native Agentform schema (.agentform) files.
"""

from agentform_compiler.agentform_ast import MergeError, ModuleBlock, merge_agentform_files
from agentform_compiler.agentform_module_loader import LoadedModule, ModuleLoader, ModuleLoadError
from agentform_compiler.agentform_module_resolver import (
    ModuleResolutionError,
    ModuleResolver,
    ResolvedModule,
    is_git_url,
    resolve_module_source,
)
from agentform_compiler.agentform_normalizer import NormalizationError, normalize_agentform

# Agentform native schema support
from agentform_compiler.agentform_parser import (
    AgentformParseError,
    discover_agentform_files,
    parse_agentform,
    parse_agentform_directory,
    parse_agentform_file,
)
from agentform_compiler.agentform_resolver import (
    ResolutionError,
    ResolutionResult,
    resolve_references,
)
from agentform_compiler.agentform_validator import (
    AgentformValidationError,
    AgentformValidationResult,
    validate_agentform,
)
from agentform_compiler.compiler import (
    CompilationError,
    compile_agentform,
    compile_agentform_directory,
    compile_agentform_file,
    compile_file,
    validate_agentform_directory,
    validate_agentform_file,
    validate_file,
)
from agentform_compiler.credentials import (
    CredentialError,
    get_env_var_name,
    is_env_reference,
    resolve_env_var,
)
from agentform_compiler.ir_generator import IRGenerationError, generate_ir
from agentform_compiler.validator import ValidationError, ValidationResult, validate_spec

__all__ = [
    "AgentformParseError",
    "AgentformValidationError",
    "AgentformValidationResult",
    "CompilationError",
    "CredentialError",
    "IRGenerationError",
    "LoadedModule",
    "MergeError",
    "ModuleBlock",
    "ModuleLoadError",
    "ModuleLoader",
    "ModuleResolutionError",
    "ModuleResolver",
    "NormalizationError",
    "ResolutionError",
    "ResolutionResult",
    "ResolvedModule",
    "ValidationError",
    "ValidationResult",
    "compile_agentform",
    "compile_agentform_directory",
    "compile_agentform_file",
    "compile_file",
    "discover_agentform_files",
    "generate_ir",
    "get_env_var_name",
    "is_env_reference",
    "is_git_url",
    "merge_agentform_files",
    "normalize_agentform",
    "parse_agentform",
    "parse_agentform_directory",
    "parse_agentform_file",
    "resolve_env_var",
    "resolve_module_source",
    "resolve_references",
    "validate_agentform",
    "validate_agentform_directory",
    "validate_agentform_file",
    "validate_file",
    "validate_spec",
]
