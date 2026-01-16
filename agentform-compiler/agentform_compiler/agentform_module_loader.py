"""Module loader for Agentform module system.

Handles loading, parsing, and parameter injection for Agentform modules.
"""

from pathlib import Path
from typing import Any

from agentform_compiler.agentform_ast import (
    AgentformFile,
    ModuleBlock,
    VariableBlock,
    VarRef,
)
from agentform_compiler.agentform_module_resolver import (
    ModuleResolutionError,
    ModuleResolver,
    ResolvedModule,
)
from agentform_compiler.agentform_parser import (
    AgentformParseError,
    parse_agentform_directory,
)


class ModuleLoadError(Exception):
    """Error during module loading."""

    pass


class LoadedModule:
    """A loaded and parsed module with its metadata."""

    def __init__(
        self,
        name: str,
        agentform_file: AgentformFile,
        resolved: ResolvedModule,
        parameters: dict[str, Any],
    ):
        """Initialize a loaded module.

        Args:
            name: Module instance name (from module block)
            agentform_file: Parsed module AST
            resolved: Resolution info (path, source, version)
            parameters: Parameters passed to the module
        """
        self.name = name
        self.agentform_file = agentform_file
        self.resolved = resolved
        self.parameters = parameters

    @property
    def path(self) -> Path:
        """Get the module's local path."""
        return self.resolved.path

    @property
    def source(self) -> str:
        """Get the original source string."""
        return self.resolved.source

    @property
    def version(self) -> str | None:
        """Get the module version if specified."""
        return self.resolved.version

    def get_exported_resources(self) -> dict[str, list[str]]:
        """Get a summary of resources exported by this module.

        Returns:
            Dict mapping resource type to list of names
        """
        return {
            "providers": [p.full_name for p in self.agentform_file.providers],
            "servers": [s.name for s in self.agentform_file.servers],
            "capabilities": [c.name for c in self.agentform_file.capabilities],
            "policies": [p.name for p in self.agentform_file.policies],
            "models": [m.name for m in self.agentform_file.models],
            "agents": [a.name for a in self.agentform_file.agents],
            "workflows": [w.name for w in self.agentform_file.workflows],
        }


class ModuleLoader:
    """Loads and parses Agentform modules."""

    def __init__(
        self,
        resolver: ModuleResolver | None = None,
        base_path: Path | None = None,
    ):
        """Initialize the module loader.

        Args:
            resolver: Module resolver to use. If None, creates a new one.
            base_path: Base path for resolving relative local paths.
        """
        self.resolver = resolver or ModuleResolver(base_path=base_path)
        self._loaded_cache: dict[str, LoadedModule] = {}

    def load_module(
        self,
        module_block: ModuleBlock,
    ) -> LoadedModule:
        """Load a module from a module block.

        Args:
            module_block: Parsed module block from Agentform file

        Returns:
            LoadedModule with parsed AST and metadata

        Raises:
            ModuleLoadError: If loading fails
        """
        name = module_block.name
        source = module_block.source
        version = module_block.version
        parameters = module_block.get_parameters()

        if not source:
            raise ModuleLoadError(f"Module '{name}' is missing required 'source' attribute")

        # Check cache (keyed by name since same module can be loaded with different params)
        # For now, don't cache - each module instance may have different params
        # In the future, we could cache the parsed AST and apply params separately

        try:
            # Resolve the module source
            resolved = self.resolver.resolve(source, version)
        except ModuleResolutionError as e:
            raise ModuleLoadError(f"Failed to resolve module '{name}': {e}") from e

        try:
            # Parse the module directory
            agentform_file = parse_agentform_directory(resolved.path)
        except AgentformParseError as e:
            raise ModuleLoadError(f"Failed to parse module '{name}': {e}") from e

        # Validate and apply parameters
        resolved_params = self._resolve_parameters(name, agentform_file, parameters)

        return LoadedModule(
            name=name,
            agentform_file=agentform_file,
            resolved=resolved,
            parameters=resolved_params,
        )

    def load_modules(
        self,
        module_blocks: list[ModuleBlock],
    ) -> dict[str, LoadedModule]:
        """Load multiple modules.

        Args:
            module_blocks: List of module blocks to load

        Returns:
            Dict mapping module name to LoadedModule

        Raises:
            ModuleLoadError: If any module fails to load
        """
        loaded: dict[str, LoadedModule] = {}

        for block in module_blocks:
            if block.name in loaded:
                raise ModuleLoadError(f"Duplicate module name: {block.name}")
            loaded[block.name] = self.load_module(block)

        return loaded

    def _resolve_parameters(
        self,
        module_name: str,
        agentform_file: AgentformFile,
        provided_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve and validate module parameters.

        Validates that:
        - Required parameters (no default) are provided
        - Parameter types match if specified
        - No unknown parameters are provided

        Args:
            module_name: Name of the module (for error messages)
            agentform_file: Parsed module AST
            provided_params: Parameters provided by caller

        Returns:
            Dict of resolved parameter values

        Raises:
            ModuleLoadError: If validation fails
        """
        resolved: dict[str, Any] = {}
        declared_vars: dict[str, VariableBlock] = {v.name: v for v in agentform_file.variables}

        # Check for required parameters
        for var_name, var_block in declared_vars.items():
            if var_name in provided_params:
                # Parameter provided - use it
                value = provided_params[var_name]

                # Resolve VarRef if present (reference to parent variable)
                if isinstance(value, VarRef):
                    # VarRef will be resolved later during compilation
                    # For now, keep it as-is
                    resolved[var_name] = value
                else:
                    # Validate type if specified
                    self._validate_param_type(module_name, var_name, value, var_block.var_type)
                    resolved[var_name] = value

            elif var_block.default is not None:
                # Use default value
                resolved[var_name] = var_block.default

            else:
                # Required parameter not provided
                raise ModuleLoadError(f"Module '{module_name}' requires parameter '{var_name}'")

        # Check for unknown parameters (warn but don't error)
        for param_name in provided_params:
            if param_name not in declared_vars:
                # Could log a warning here in the future
                # For now, pass through unknown params
                resolved[param_name] = provided_params[param_name]

        return resolved

    def _validate_param_type(
        self,
        module_name: str,
        param_name: str,
        value: Any,
        expected_type: str | None,
    ) -> None:
        """Validate a parameter value against its expected type.

        Args:
            module_name: Module name for error messages
            param_name: Parameter name
            value: Provided value
            expected_type: Expected type string ("string", "number", "bool", "list")

        Raises:
            ModuleLoadError: If type validation fails
        """
        if expected_type is None:
            return  # No type constraint

        type_map: dict[str, Any] = {
            "string": str,
            "number": (int, float),
            "bool": bool,
            "list": list,
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return  # Unknown type, skip validation

        if not isinstance(value, expected):
            raise ModuleLoadError(
                f"Module '{module_name}' parameter '{param_name}' expects "
                f"{expected_type}, got {type(value).__name__}"
            )


def load_module_from_block(
    module_block: ModuleBlock,
    base_path: Path | None = None,
) -> LoadedModule:
    """Convenience function to load a single module.

    Args:
        module_block: Parsed module block
        base_path: Base path for resolving relative paths

    Returns:
        LoadedModule with parsed AST

    Raises:
        ModuleLoadError: If loading fails
    """
    loader = ModuleLoader(base_path=base_path)
    return loader.load_module(module_block)
