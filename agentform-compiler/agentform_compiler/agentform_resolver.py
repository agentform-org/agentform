"""Reference resolver for Agentform native schema.

Builds a symbol table from parsed blocks and resolves all references
to ensure they point to valid targets.
"""

from dataclasses import dataclass, field

from agentform_compiler.agentform_ast import (
    AgentBlock,
    AgentformFile,
    CapabilityBlock,
    ModelBlock,
    NestedBlock,
    ProviderBlock,
    Reference,
    SourceLocation,
    StepBlock,
    VarRef,
    WorkflowBlock,
)


@dataclass
class Symbol:
    """A symbol in the symbol table."""

    name: str
    kind: str  # "provider", "model", "agent", "policy", "workflow", "step", "server", "capability", "variable", "module"
    location: SourceLocation | None = None
    parent: str | None = None  # For nested symbols (e.g., steps belong to workflows)
    block: object | None = None  # Reference to the actual block
    module: str | None = None  # Module name if this symbol belongs to a module


@dataclass
class ResolutionError:
    """An error during reference resolution."""

    message: str
    location: SourceLocation | None = None

    def __str__(self) -> str:
        if self.location:
            return f"{self.location}: {self.message}"
        return self.message


@dataclass
class ResolutionResult:
    """Result of reference resolution."""

    symbols: dict[str, Symbol] = field(default_factory=dict)
    errors: list[ResolutionError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if resolution succeeded without errors."""
        return len(self.errors) == 0

    def add_error(self, message: str, location: SourceLocation | None = None) -> None:
        """Add a resolution error."""
        self.errors.append(ResolutionError(message, location))

    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the table."""
        self.symbols[symbol.name] = symbol


class ReferenceResolver:
    """Resolves references in an Agentform AST.

    This class:
    1. Builds a symbol table from all named blocks
    2. Resolves all references to ensure they point to valid targets
    3. Reports unresolved references with source locations
    """

    def __init__(self, agentform_file: AgentformFile):
        self.agentform_file = agentform_file
        self.result = ResolutionResult()

    def resolve(self) -> ResolutionResult:
        """Resolve all references in the AST.

        Returns:
            ResolutionResult with symbol table and any errors
        """
        # Phase 1: Build symbol table
        self._build_symbol_table()

        # Phase 2: Resolve all references
        self._resolve_references()

        return self.result

    def _build_symbol_table(self) -> None:
        """Build the symbol table from all named blocks."""
        # Variables: var.name
        for variable in self.agentform_file.variables:
            full_name = f"var.{variable.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate variable: {variable.name}",
                    variable.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="variable",
                        location=variable.location,
                        block=variable,
                    )
                )

        # Providers: provider.llm.openai.default
        for provider in self.agentform_file.providers:
            full_name = f"provider.{provider.full_name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate provider: {full_name}",
                    provider.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="provider",
                        location=provider.location,
                        block=provider,
                    )
                )

        # Servers: server.name
        for server in self.agentform_file.servers:
            full_name = f"server.{server.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate server: {server.name}",
                    server.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="server",
                        location=server.location,
                        block=server,
                    )
                )

        # Capabilities: capability.name
        for capability in self.agentform_file.capabilities:
            full_name = f"capability.{capability.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate capability: {capability.name}",
                    capability.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="capability",
                        location=capability.location,
                        block=capability,
                    )
                )

        # Policies: policy.name
        for policy in self.agentform_file.policies:
            full_name = f"policy.{policy.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate policy: {policy.name}",
                    policy.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="policy",
                        location=policy.location,
                        block=policy,
                    )
                )

        # Models: model.name
        for model in self.agentform_file.models:
            full_name = f"model.{model.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate model: {model.name}",
                    model.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="model",
                        location=model.location,
                        block=model,
                    )
                )

        # Agents: agent.name
        for agent in self.agentform_file.agents:
            full_name = f"agent.{agent.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate agent: {agent.name}",
                    agent.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="agent",
                        location=agent.location,
                        block=agent,
                    )
                )

        # Workflows and their steps
        for workflow in self.agentform_file.workflows:
            workflow_name = f"workflow.{workflow.name}"
            if workflow_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate workflow: {workflow.name}",
                    workflow.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=workflow_name,
                        kind="workflow",
                        location=workflow.location,
                        block=workflow,
                    )
                )

            # Steps are scoped to workflow but we track them globally with step.id
            # for reference resolution (used within the same workflow)
            step_ids: set[str] = set()
            for step in workflow.steps:
                if step.step_id in step_ids:
                    self.result.add_error(
                        f"Duplicate step '{step.step_id}' in workflow '{workflow.name}'",
                        step.location,
                    )
                else:
                    step_ids.add(step.step_id)
                    # Register step with workflow-scoped name
                    step_name = f"step.{step.step_id}"
                    self.result.add_symbol(
                        Symbol(
                            name=step_name,
                            kind="step",
                            location=step.location,
                            parent=workflow.name,
                            block=step,
                        )
                    )

        # Modules: module.name
        for module in self.agentform_file.modules:
            full_name = f"module.{module.name}"
            if full_name in self.result.symbols:
                self.result.add_error(
                    f"Duplicate module: {module.name}",
                    module.location,
                )
            else:
                self.result.add_symbol(
                    Symbol(
                        name=full_name,
                        kind="module",
                        location=module.location,
                        block=module,
                    )
                )

    def _resolve_references(self) -> None:
        """Resolve all references in the AST."""
        # Resolve provider references (variable references in api_key)
        for provider in self.agentform_file.providers:
            self._resolve_provider_references(provider)

        # Resolve model references (provider)
        for model in self.agentform_file.models:
            self._resolve_model_references(model)

        # Resolve agent references (model, policy, capabilities)
        for agent in self.agentform_file.agents:
            self._resolve_agent_references(agent)

        # Resolve capability references (server)
        for capability in self.agentform_file.capabilities:
            self._resolve_capability_references(capability)

        # Resolve workflow references (steps, agents)
        for workflow in self.agentform_file.workflows:
            self._resolve_workflow_references(workflow)

    def _resolve_provider_references(self, provider: ProviderBlock) -> None:
        """Resolve references in a provider block."""
        # Check api_key variable reference
        api_key_attr = provider.get_attribute("api_key")
        if isinstance(api_key_attr, VarRef):
            self._check_var_ref(api_key_attr, provider.location)

        # Check nested blocks for variable references
        for block in provider.blocks:
            self._resolve_nested_block_var_refs(block)

    def _resolve_model_references(self, model: ModelBlock) -> None:
        """Resolve references in a model block."""
        # Check provider reference
        provider_attr = model.get_attribute("provider")
        if isinstance(provider_attr, Reference):
            self._check_reference(provider_attr, "provider", model.location)

    def _resolve_agent_references(self, agent: AgentBlock) -> None:
        """Resolve references in an agent block."""
        # Check model reference
        model_attr = agent.get_attribute("model")
        if isinstance(model_attr, Reference):
            self._check_reference(model_attr, "model", agent.location)

        # Check fallback_models array
        fallback_attr = agent.get_attribute("fallback_models")
        if isinstance(fallback_attr, list):
            for item in fallback_attr:
                if isinstance(item, Reference):
                    self._check_reference(item, "model", agent.location)

        # Check policy reference
        policy_attr = agent.get_attribute("policy")
        if isinstance(policy_attr, Reference):
            self._check_reference(policy_attr, "policy", agent.location)

        # Check allow array (capabilities)
        allow_attr = agent.get_attribute("allow")
        if isinstance(allow_attr, list):
            for item in allow_attr:
                if isinstance(item, Reference):
                    self._check_reference(item, "capability", agent.location)

    def _resolve_capability_references(self, capability: CapabilityBlock) -> None:
        """Resolve references in a capability block."""
        # Check server reference
        server_attr = capability.get_attribute("server")
        if isinstance(server_attr, Reference):
            self._check_reference(server_attr, "server", capability.location)

    def _resolve_workflow_references(self, workflow: WorkflowBlock) -> None:
        """Resolve references in a workflow block."""
        # Check entry reference
        entry_attr = workflow.get_attribute("entry")
        if isinstance(entry_attr, Reference):
            self._check_reference(entry_attr, "step", workflow.location)

        # Check step references
        for step in workflow.steps:
            self._resolve_step_references(step, workflow.name)

    def _resolve_step_references(self, step: StepBlock, workflow_name: str) -> None:
        """Resolve references in a step block."""
        # Check agent reference
        agent_attr = step.get_attribute("agent")
        if isinstance(agent_attr, Reference):
            self._check_reference(agent_attr, "agent", step.location)

        # Check capability reference (for call steps)
        cap_attr = step.get_attribute("capability")
        if isinstance(cap_attr, Reference):
            self._check_reference(cap_attr, "capability", step.location)

        # Check next reference
        next_attr = step.get_attribute("next")
        if isinstance(next_attr, Reference):
            self._check_reference(next_attr, "step", step.location)

        # Check on_true/on_false references (for condition steps)
        for attr_name in ["on_true", "on_false", "on_approve", "on_reject"]:
            attr_val = step.get_attribute(attr_name)
            if isinstance(attr_val, Reference):
                self._check_reference(attr_val, "step", step.location)

        # Check input block references
        input_block = step.get_input_block()
        if input_block:
            self._resolve_nested_block_values(input_block)

        # Check args block references
        args_block = step.get_args_block()
        if args_block:
            self._resolve_nested_block_values(args_block)

    def _resolve_nested_block_values(self, block: NestedBlock) -> None:
        """Resolve references in nested block values.

        Note: input.* and result.* references are runtime values,
        not compile-time references to blocks, so we don't validate them.
        """
        for attr in block.attributes:
            value = attr.value
            if isinstance(value, Reference):
                # input.* and result.* and state.* are runtime references
                if value.parts[0] in ("input", "result", "state"):
                    continue
                # Other references should be validated
                self._check_reference_exists(value)
            elif isinstance(value, VarRef):
                self._check_var_ref(value, attr.location)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, VarRef):
                        self._check_var_ref(item, attr.location)

    def _check_reference(
        self,
        ref: Reference,
        expected_kind: str,
        context_location: SourceLocation | None,
    ) -> None:
        """Check that a reference exists and has the expected kind."""
        ref_path = ref.path

        # Module references (module.name.*) are validated after module loading
        # For now, just check that the module exists
        if ref.parts[0] == "module" and len(ref.parts) >= 2:
            module_name = f"module.{ref.parts[1]}"
            if module_name not in self.result.symbols:
                self.result.add_error(
                    f"Unresolved module reference: {module_name}",
                    ref.location or context_location,
                )
            # Module resource references will be validated after module loading
            return

        symbol = self.result.symbols.get(ref_path)

        if symbol is None:
            self.result.add_error(
                f"Unresolved reference: {ref_path}",
                ref.location or context_location,
            )
        elif symbol.kind != expected_kind:
            self.result.add_error(
                f"Reference '{ref_path}' is a {symbol.kind}, expected {expected_kind}",
                ref.location or context_location,
            )

    def _check_reference_exists(self, ref: Reference) -> None:
        """Check that a reference exists (any kind)."""
        ref_path = ref.path

        # Module references (module.name.*) are validated after module loading
        if ref.parts[0] == "module" and len(ref.parts) >= 2:
            module_name = f"module.{ref.parts[1]}"
            if module_name not in self.result.symbols:
                self.result.add_error(
                    f"Unresolved module reference: {module_name}",
                    ref.location,
                )
            return

        if ref_path not in self.result.symbols:
            self.result.add_error(
                f"Unresolved reference: {ref_path}",
                ref.location,
            )

    def _check_var_ref(
        self,
        var_ref: VarRef,
        context_location: SourceLocation | None,
    ) -> None:
        """Check that a variable reference points to a declared variable."""
        ref_path = f"var.{var_ref.var_name}"
        symbol = self.result.symbols.get(ref_path)

        if symbol is None:
            self.result.add_error(
                f"Unresolved variable reference: {ref_path}",
                var_ref.location or context_location,
            )
        elif symbol.kind != "variable":
            self.result.add_error(
                f"Reference '{ref_path}' is a {symbol.kind}, expected variable",
                var_ref.location or context_location,
            )

    def _resolve_nested_block_var_refs(self, block: NestedBlock) -> None:
        """Resolve variable references in nested blocks."""
        for attr in block.attributes:
            if isinstance(attr.value, VarRef):
                self._check_var_ref(attr.value, attr.location)
            elif isinstance(attr.value, list):
                for item in attr.value:
                    if isinstance(item, VarRef):
                        self._check_var_ref(item, attr.location)

        # Recursively check nested blocks
        for nested in block.blocks:
            self._resolve_nested_block_var_refs(nested)


def resolve_references(agentform_file: AgentformFile) -> ResolutionResult:
    """Resolve all references in an Agentform file.

    Args:
        agentform_file: Parsed Agentform AST

    Returns:
        ResolutionResult with symbol table and any errors
    """
    resolver = ReferenceResolver(agentform_file)
    return resolver.resolve()


def add_module_symbols(
    result: ResolutionResult,
    module_name: str,
    module_agentform: AgentformFile,
) -> None:
    """Add symbols from a loaded module to the resolution result.

    Module symbols are namespaced as:
        module.<module_name>.<resource_type>.<resource_name>

    Args:
        result: Resolution result to add symbols to
        module_name: Name of the module instance
        module_agentform: Parsed module AgentformFile
    """
    # Add provider symbols
    for provider in module_agentform.providers:
        full_name = f"module.{module_name}.provider.{provider.full_name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="provider",
                location=provider.location,
                block=provider,
                module=module_name,
            )
        )

    # Add server symbols
    for server in module_agentform.servers:
        full_name = f"module.{module_name}.server.{server.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="server",
                location=server.location,
                block=server,
                module=module_name,
            )
        )

    # Add capability symbols
    for capability in module_agentform.capabilities:
        full_name = f"module.{module_name}.capability.{capability.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="capability",
                location=capability.location,
                block=capability,
                module=module_name,
            )
        )

    # Add policy symbols
    for policy in module_agentform.policies:
        full_name = f"module.{module_name}.policy.{policy.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="policy",
                location=policy.location,
                block=policy,
                module=module_name,
            )
        )

    # Add model symbols
    for model in module_agentform.models:
        full_name = f"module.{module_name}.model.{model.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="model",
                location=model.location,
                block=model,
                module=module_name,
            )
        )

    # Add agent symbols
    for agent in module_agentform.agents:
        full_name = f"module.{module_name}.agent.{agent.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="agent",
                location=agent.location,
                block=agent,
                module=module_name,
            )
        )

    # Add workflow symbols
    for workflow in module_agentform.workflows:
        full_name = f"module.{module_name}.workflow.{workflow.name}"
        result.add_symbol(
            Symbol(
                name=full_name,
                kind="workflow",
                location=workflow.location,
                block=workflow,
                module=module_name,
            )
        )
