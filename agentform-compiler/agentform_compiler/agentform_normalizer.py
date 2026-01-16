"""Normalizer that transforms Agentform AST to SpecRoot format.

This module converts the parsed Agentform AST into the SpecRoot model
for validation and IR generation.
"""

from typing import Any

from agentform_compiler.agentform_ast import (
    AgentformFile,
    AndExpr,
    ComparisonExpr,
    ConditionalExpr,
    NestedBlock,
    NotExpr,
    OrExpr,
    Reference,
    SourceLocation,
    StateRef,
    StepBlock,
    Value,
    VariableBlock,
    VarRef,
)
from agentform_compiler.agentform_resolver import ResolutionResult
from agentform_schema.models import (
    AgentConfig,
    BudgetConfig,
    CapabilityConfig,
    LLMProviderConfig,
    LLMProviderParams,
    ModelConfig,
    PolicyConfig,
    ProjectConfig,
    ProvidersConfig,
    ServerAuthConfig,
    ServerConfig,
    SideEffect,
    SpecRoot,
    StepType,
    Transport,
    WorkflowConfig,
    WorkflowStep,
)


class NormalizationError(Exception):
    """Error during normalization."""

    def __init__(self, message: str, location: SourceLocation | None = None):
        self.location = location
        if location:
            super().__init__(f"{location}: {message}")
        else:
            super().__init__(message)


class AgentformNormalizer:
    """Transforms Agentform AST to SpecRoot."""

    def __init__(
        self,
        agentform_file: AgentformFile,
        resolution: ResolutionResult,
        variables: dict[str, Any] | None = None,
        loaded_modules: dict[str, Any] | None = None,  # dict[str, LoadedModule]
    ):
        self.agentform_file = agentform_file
        self.resolution = resolution
        self.variables = variables or {}
        self.loaded_modules = loaded_modules or {}
        # Build variable defaults from declarations
        self._variable_defs: dict[str, VariableBlock] = {
            v.name: v for v in agentform_file.variables
        }
        # Cache for resolved model info
        self._model_cache: dict[str, tuple[str, str, LLMProviderParams | None]] = {}

    def normalize(self) -> SpecRoot:
        """Normalize Agentform AST to SpecRoot.

        Returns:
            SpecRoot model compatible with existing pipeline

        Raises:
            NormalizationError: If normalization fails
        """
        # Build model cache first (maps model name -> (provider, model_id, params))
        self._build_model_cache()

        # Also build model cache for loaded modules
        for module_name, loaded_module in self.loaded_modules.items():
            self._build_module_model_cache(module_name, loaded_module)

        # Normalize main resources
        providers = self._normalize_providers()
        servers = self._normalize_servers()
        capabilities = self._normalize_capabilities()
        policies = self._normalize_policies()
        agents = self._normalize_agents()
        workflows = self._normalize_workflows()

        # Merge module resources with namespaced names
        self._merge_module_resources(providers, servers, capabilities, policies, agents, workflows)

        return SpecRoot(
            version=self._get_version(),
            project=self._normalize_project(),
            providers=providers,
            servers=servers,
            capabilities=capabilities,
            policies=policies,
            agents=agents,
            workflows=workflows,
        )

    def _build_module_model_cache(self, module_name: str, loaded_module: Any) -> None:
        """Build model cache entries for a module's models.

        Args:
            module_name: Name of the module instance
            loaded_module: LoadedModule instance
        """
        module_agentform = loaded_module.agentform_file

        for model in module_agentform.models:
            # Get provider reference
            provider_ref = model.get_attribute("provider")
            if not isinstance(provider_ref, Reference):
                continue

            # Convert provider reference to key using same logic as _normalize_providers
            provider_key = self._provider_ref_to_key(provider_ref)

            # Get model ID (resolve VarRef if needed)
            model_id = model.get_attribute("id")
            if isinstance(model_id, VarRef):
                # Resolve variable using module's parameters
                var_name = model_id.var_name
                if var_name in loaded_module.parameters:
                    model_id = str(loaded_module.parameters[var_name])
                else:
                    # Check module's variable defaults
                    for var_def in module_agentform.variables:
                        if var_def.name == var_name and var_def.default is not None:
                            model_id = str(var_def.default)
                            break
                    else:
                        model_id = model.name
            elif not isinstance(model_id, str):
                model_id = model.name

            # Get params
            params = None
            params_block = model.get_params_block()
            if params_block:
                params = self._parse_llm_params(params_block)

            # Cache with module-namespaced key
            cache_key = f"module.{module_name}.model.{model.name}"
            # Store with module-namespaced provider (matches key in providers.llm)
            namespaced_provider = f"module.{module_name}.{provider_key}"
            self._model_cache[cache_key] = (namespaced_provider, model_id, params)

    def _merge_module_resources(
        self,
        providers: ProvidersConfig,
        servers: list[ServerConfig],
        capabilities: list[CapabilityConfig],
        policies: list[PolicyConfig],
        agents: list[AgentConfig],
        workflows: list[WorkflowConfig],
    ) -> None:
        """Merge resources from loaded modules into the main spec.

        Module resources are namespaced as: module.<module_name>.<resource_name>

        Args:
            providers: Main providers config to merge into
            servers: Main servers list to merge into
            capabilities: Main capabilities list to merge into
            policies: Main policies list to merge into
            agents: Main agents list to merge into
            workflows: Main workflows list to merge into
        """
        for module_name, loaded_module in self.loaded_modules.items():
            module_agentform = loaded_module.agentform_file
            module_params = loaded_module.parameters

            # Resolve VarRefs in module parameters against parent's variables
            resolved_params: dict[str, Any] = {}
            for param_name, param_value in module_params.items():
                if isinstance(param_value, VarRef):
                    # Resolve the VarRef against parent's variables
                    resolved_value, _ = self._resolve_variable(param_value)
                    resolved_params[param_name] = resolved_value
                else:
                    resolved_params[param_name] = param_value

            # Create a sub-normalizer for the module with resolved parameters
            module_normalizer = AgentformNormalizer(
                module_agentform,
                self.resolution,
                resolved_params,
                {},  # Modules don't have nested modules (for now)
            )
            # Build model cache for the module
            module_normalizer._build_model_cache()

            # Merge providers
            module_providers = module_normalizer._normalize_providers()
            if module_providers.llm:
                for provider_name, provider_config in module_providers.llm.items():
                    namespaced_name = f"module.{module_name}.{provider_name}"
                    if providers.llm is None:
                        providers.llm = {}
                    providers.llm[namespaced_name] = provider_config

            # Merge servers
            module_servers = module_normalizer._normalize_servers()
            for server_config in module_servers:
                # Namespace the server name
                server_config.name = f"module.{module_name}.{server_config.name}"
                servers.append(server_config)

            # Merge capabilities
            module_capabilities = module_normalizer._normalize_capabilities()
            for cap_config in module_capabilities:
                # Namespace the capability name and server reference
                cap_config.name = f"module.{module_name}.{cap_config.name}"
                if cap_config.server:
                    cap_config.server = f"module.{module_name}.{cap_config.server}"
                capabilities.append(cap_config)

            # Merge policies
            module_policies = module_normalizer._normalize_policies()
            for policy_config in module_policies:
                # Namespace the policy name
                policy_config.name = f"module.{module_name}.{policy_config.name}"
                policies.append(policy_config)

            # Merge agents (but update their model/policy/capability references)
            module_agents = module_normalizer._normalize_agents()
            for agent_config in module_agents:
                # Namespace the agent name
                agent_config.name = f"module.{module_name}.{agent_config.name}"
                # Update provider reference to be namespaced
                if agent_config.provider and not agent_config.provider.startswith("module."):
                    agent_config.provider = f"module.{module_name}.{agent_config.provider}"
                # Update policy reference if it's from the module
                if agent_config.policy and not agent_config.policy.startswith("module."):
                    agent_config.policy = f"module.{module_name}.{agent_config.policy}"
                # Update capability references in allow list
                if agent_config.allow:
                    agent_config.allow = [
                        f"module.{module_name}.{cap}" if not cap.startswith("module.") else cap
                        for cap in agent_config.allow
                    ]
                agents.append(agent_config)

            # Merge workflows (namespaced)
            module_workflows = module_normalizer._normalize_workflows()
            for workflow_config in module_workflows:
                # Namespace the workflow name
                workflow_config.name = f"module.{module_name}.{workflow_config.name}"
                # Update agent and capability references in workflow steps
                for step in workflow_config.steps:
                    if step.agent and not step.agent.startswith("module."):
                        step.agent = f"module.{module_name}.{step.agent}"
                    if step.capability and not step.capability.startswith("module."):
                        step.capability = f"module.{module_name}.{step.capability}"
                workflows.append(workflow_config)

    def _get_version(self) -> str:
        """Get version from agentform block."""
        if self.agentform_file.agentform and self.agentform_file.agentform.version:
            return self.agentform_file.agentform.version
        return "0.1"

    def _normalize_project(self) -> ProjectConfig:
        """Normalize project configuration."""
        name = "unnamed"
        if self.agentform_file.agentform and self.agentform_file.agentform.project:
            name = self.agentform_file.agentform.project
        return ProjectConfig(name=name)

    def _resolve_variable(self, var_ref: VarRef) -> tuple[str, bool]:
        """Resolve a variable reference to its value.

        Args:
            var_ref: Variable reference to resolve

        Returns:
            Tuple of (resolved_value, is_sensitive)

        Raises:
            NormalizationError: If variable is not declared or value not provided
        """
        var_name = var_ref.var_name
        var_def = self._variable_defs.get(var_name)

        if var_def is None:
            raise NormalizationError(
                f"Undefined variable: {var_name}",
                var_ref.location,
            )

        # Check if value is provided at runtime
        if var_name in self.variables:
            value = self.variables[var_name]
            return str(value) if not isinstance(value, str) else value, var_def.sensitive

        # Use default if available
        if var_def.default is not None:
            value = var_def.default
            return str(value) if not isinstance(value, str) else value, var_def.sensitive

        # No value provided and no default
        raise NormalizationError(
            f"Variable '{var_name}' has no value (no default and not provided at runtime)",
            var_ref.location,
        )

    def _normalize_providers(self) -> ProvidersConfig:
        """Normalize provider blocks to ProvidersConfig."""
        llm_providers: dict[str, LLMProviderConfig] = {}

        for provider in self.agentform_file.providers:
            # Parse provider type (e.g., "llm.openai" -> category="llm", vendor="openai")
            parts = provider.provider_type.split(".")
            if len(parts) >= 2 and parts[0] == "llm":
                vendor = ".".join(parts[1:])  # Handle nested vendor names
                # Create unique key: vendor_name or just vendor if name is "default"
                key = f"{vendor}" if provider.name == "default" else f"{vendor}_{provider.name}"

                # Get api_key
                api_key_val = provider.get_attribute("api_key")
                if isinstance(api_key_val, VarRef):
                    api_key, _ = self._resolve_variable(api_key_val)
                else:
                    api_key = str(api_key_val) if api_key_val else ""

                # Get default_params from nested block
                default_params = None
                for block in provider.blocks:
                    if block.block_type == "default_params":
                        default_params = self._parse_llm_params(block)
                        break

                llm_providers[key] = LLMProviderConfig(
                    api_key=api_key,
                    default_params=default_params,
                )

        return ProvidersConfig(llm=llm_providers)

    def _normalize_servers(self) -> list[ServerConfig]:
        """Normalize server blocks."""
        servers: list[ServerConfig] = []

        for server in self.agentform_file.servers:
            # Get command
            command = server.get_attribute("command")
            if not isinstance(command, list):
                command = []
            command_strs = [self._value_to_str(c) for c in command]

            # Get transport
            transport_val = server.get_attribute("transport")
            transport = Transport.STDIO
            if transport_val == "stdio":
                transport = Transport.STDIO

            # Get type
            server_type = server.get_attribute("type")
            if not isinstance(server_type, str):
                server_type = "mcp"

            # Get auth if present
            auth = None
            for block in server.blocks:
                if block.block_type == "auth":
                    token = block.get_attribute("token")
                    if isinstance(token, VarRef):
                        resolved_token, _ = self._resolve_variable(token)
                        auth = ServerAuthConfig(token=resolved_token)
                    elif isinstance(token, str):
                        auth = ServerAuthConfig(token=token)
                    break

            servers.append(
                ServerConfig(
                    name=server.name,
                    type=server_type,
                    transport=transport,
                    command=command_strs,
                    auth=auth,
                )
            )

        return servers

    def _normalize_capabilities(self) -> list[CapabilityConfig]:
        """Normalize capability blocks."""
        capabilities: list[CapabilityConfig] = []

        for cap in self.agentform_file.capabilities:
            # Get server reference
            server_ref = cap.get_attribute("server")
            server_name = self._ref_to_name(server_ref, "server")
            if not server_name:
                raise NormalizationError(
                    f"Capability '{cap.name}' must have a server reference",
                    cap.location,
                )

            # Get method
            method = cap.get_attribute("method")
            if not isinstance(method, str):
                method = cap.name

            # Get side_effect
            side_effect_val = cap.get_attribute("side_effect")
            side_effect = SideEffect.READ
            if side_effect_val == "write":
                side_effect = SideEffect.WRITE

            # Get requires_approval
            requires_approval = cap.get_attribute("requires_approval")
            if not isinstance(requires_approval, bool):
                requires_approval = False

            capabilities.append(
                CapabilityConfig(
                    name=cap.name,
                    server=server_name,
                    method=method,
                    side_effect=side_effect,
                    requires_approval=requires_approval,
                )
            )

        return capabilities

    def _normalize_policies(self) -> list[PolicyConfig]:
        """Normalize policy blocks."""
        policies: list[PolicyConfig] = []

        for policy in self.agentform_file.policies:
            # Merge all budget blocks
            budgets = BudgetConfig()
            for budget_block in policy.get_budgets_blocks():
                for attr in budget_block.attributes:
                    if attr.name == "max_cost_usd_per_run" and isinstance(attr.value, (int, float)):
                        budgets.max_cost_usd_per_run = float(attr.value)
                    elif attr.name == "max_capability_calls" and isinstance(attr.value, int):
                        budgets.max_capability_calls = attr.value
                    elif attr.name == "timeout_seconds" and isinstance(attr.value, int):
                        budgets.timeout_seconds = attr.value

            policies.append(
                PolicyConfig(
                    name=policy.name,
                    budgets=budgets,
                )
            )

        return policies

    def _build_model_cache(self) -> None:
        """Build cache of model info for agent normalization."""
        for model in self.agentform_file.models:
            # Get provider reference
            provider_ref = model.get_attribute("provider")
            provider_name = self._provider_ref_to_key(provider_ref)

            # Get model id (resolve VarRef if needed)
            model_id = model.get_attribute("id")
            if isinstance(model_id, VarRef):
                resolved_id, _ = self._resolve_variable(model_id)
                model_id = resolved_id
            elif not isinstance(model_id, str):
                model_id = model.name

            # Get params
            params = None
            params_block = model.get_params_block()
            if params_block:
                params = self._parse_llm_params(params_block)

            self._model_cache[model.name] = (provider_name, model_id, params)

    def _normalize_agents(self) -> list[AgentConfig]:
        """Normalize agent blocks."""
        agents: list[AgentConfig] = []

        for agent in self.agentform_file.agents:
            # Get primary model reference
            model_ref = agent.get_attribute("model")
            model_name = self._ref_to_name(model_ref, "model")
            if not model_name:
                raise NormalizationError(
                    f"Agent '{agent.name}' must have a model reference",
                    agent.location,
                )
            model_info = self._model_cache.get(model_name, (None, None, None))
            provider_name, model_preference, model_params = model_info

            if not provider_name or not model_preference:
                raise NormalizationError(
                    f"Could not resolve model '{model_name}' for agent '{agent.name}'",
                    agent.location,
                )

            # Get fallback model
            fallback_models = agent.get_attribute("fallback_models")
            model_fallback = None
            if isinstance(fallback_models, list) and fallback_models:
                first_fallback = fallback_models[0]
                fallback_name = self._ref_to_name(first_fallback, "model")
                if fallback_name:
                    fallback_info = self._model_cache.get(fallback_name)
                    if fallback_info:
                        model_fallback = fallback_info[1]  # model_id

            # Get instructions
            instructions = agent.get_attribute("instructions")
            if not isinstance(instructions, str):
                instructions = ""

            # Get policy reference
            policy_ref = agent.get_attribute("policy")
            policy_name = self._ref_to_name(policy_ref, "policy") if policy_ref else None

            # Get allowed capabilities
            allow_val = agent.get_attribute("allow")
            allow: list[str] = []
            if isinstance(allow_val, list):
                for item in allow_val:
                    cap_name = self._ref_to_name(item, "capability")
                    if cap_name:
                        allow.append(cap_name)

            # Get agent-specific params (override model params)
            agent_params = None
            for block in agent.blocks:
                if block.block_type == "params":
                    agent_params = self._parse_llm_params(block)
                    break

            # Merge params: model params + agent params
            final_params = model_params
            if agent_params:
                if final_params:
                    # Merge: agent params override model params
                    merged = LLMProviderParams(
                        temperature=agent_params.temperature or final_params.temperature,
                        max_tokens=agent_params.max_tokens or final_params.max_tokens,
                        top_p=agent_params.top_p or final_params.top_p,
                    )
                    final_params = merged
                else:
                    final_params = agent_params

            agents.append(
                AgentConfig(
                    name=agent.name,
                    provider=provider_name,
                    model=ModelConfig(
                        preference=model_preference,
                        fallback=model_fallback,
                    ),
                    params=final_params,
                    instructions=instructions,
                    allow=allow,
                    policy=policy_name,
                )
            )

        return agents

    def _normalize_workflows(self) -> list[WorkflowConfig]:
        """Normalize workflow blocks."""
        workflows: list[WorkflowConfig] = []

        for workflow in self.agentform_file.workflows:
            # Get entry step
            entry_ref = workflow.get_attribute("entry")
            entry = self._ref_to_name(entry_ref, "step")
            if not entry:
                entry = workflow.steps[0].step_id if workflow.steps else "start"

            # Normalize steps
            steps = [self._normalize_step(step) for step in workflow.steps]

            workflows.append(
                WorkflowConfig(
                    name=workflow.name,
                    entry=entry,
                    steps=steps,
                )
            )

        return workflows

    def _normalize_step(self, step: StepBlock) -> WorkflowStep:
        """Normalize a workflow step."""
        # Get type
        type_val = step.get_attribute("type")
        step_type = self._parse_step_type(type_val)

        # Get agent reference (for llm steps)
        agent_ref = step.get_attribute("agent")
        agent_name = self._ref_to_name(agent_ref, "agent") if agent_ref else None

        # Get capability reference (for call steps)
        cap_ref = step.get_attribute("capability")
        capability = self._ref_to_name(cap_ref, "capability") if cap_ref else None

        # Get input mapping
        input_mapping: dict[str, Any] | None = None
        input_block = step.get_input_block()
        if input_block:
            input_mapping = self._nested_block_to_dict(input_block)

        # Get args mapping (for call steps)
        args_mapping: dict[str, Any] | None = None
        args_block = step.get_args_block()
        if args_block:
            args_mapping = self._nested_block_to_dict(args_block)

        # Get save_as from output blocks
        save_as = None
        output_blocks = step.get_output_blocks()
        if output_blocks:
            # Use the label of the first output block
            save_as = output_blocks[0].label

        # Get next step
        next_ref = step.get_attribute("next")
        next_step = self._ref_to_name(next_ref, "step") if next_ref else None

        # Get condition (for condition steps)
        condition_val = step.get_attribute("condition")
        condition: str | None = None
        if isinstance(condition_val, str):
            condition = condition_val
        elif isinstance(
            condition_val, (ConditionalExpr, ComparisonExpr, AndExpr, OrExpr, NotExpr, StateRef)
        ):
            # Convert expression AST to string for runtime evaluation
            condition = self._expr_to_string(condition_val)

        # Get branching refs
        on_true_ref = step.get_attribute("on_true")
        on_true = self._ref_to_name(on_true_ref, "step") if on_true_ref else None

        on_false_ref = step.get_attribute("on_false")
        on_false = self._ref_to_name(on_false_ref, "step") if on_false_ref else None

        # Get human_approval refs
        on_approve_ref = step.get_attribute("on_approve")
        on_approve = self._ref_to_name(on_approve_ref, "step") if on_approve_ref else None

        on_reject_ref = step.get_attribute("on_reject")
        on_reject = self._ref_to_name(on_reject_ref, "step") if on_reject_ref else None

        # Get payload
        payload = step.get_attribute("payload")
        if not isinstance(payload, str):
            payload = None

        return WorkflowStep(
            id=step.step_id,
            type=step_type,
            agent=agent_name,
            input=input_mapping,
            capability=capability,
            args=args_mapping,
            condition=condition,
            on_true=on_true,
            on_false=on_false,
            payload=payload,
            on_approve=on_approve,
            on_reject=on_reject,
            save_as=save_as,
            next=next_step,
        )

    def _parse_step_type(self, type_val: Any) -> StepType:
        """Parse step type string to StepType enum."""
        if not isinstance(type_val, str):
            return StepType.END

        type_map = {
            "llm": StepType.LLM,
            "call": StepType.CALL,
            "tool": StepType.CALL,  # Alias
            "condition": StepType.CONDITION,
            "router": StepType.CONDITION,  # Alias
            "human_approval": StepType.HUMAN_APPROVAL,
            "end": StepType.END,
        }
        return type_map.get(type_val, StepType.END)

    def _parse_llm_params(self, block: NestedBlock) -> LLMProviderParams:
        """Parse LLM parameters from a nested block."""
        temperature = None
        max_tokens = None
        top_p = None

        for attr in block.attributes:
            if attr.name == "temperature" and isinstance(attr.value, (int, float)):
                temperature = float(attr.value)
            elif attr.name == "max_tokens" and isinstance(attr.value, int):
                max_tokens = attr.value
            elif attr.name == "top_p" and isinstance(attr.value, (int, float)):
                top_p = float(attr.value)

        return LLMProviderParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    def _ref_to_name(self, ref: Any, expected_prefix: str) -> str | None:
        """Extract name from a reference, removing the type prefix.

        Handles:
        - Direct references: policy.default -> default
        - Module references: module.llm.policy.standard -> module.llm.standard
        - Module model references: module.llm.model.default -> module.llm.model.default
          (kept with "model." for model cache lookup)
        """
        if isinstance(ref, Reference):
            parts = ref.parts
            # Handle module references: module.<name>.<type>.<resource>
            if parts[0] == "module" and len(parts) >= 4:
                module_name = parts[1]
                resource_type = parts[2]
                resource_name = ".".join(parts[3:])
                # For models, keep the full path for cache lookup
                if resource_type == "model":
                    return f"module.{module_name}.model.{resource_name}"
                # For other resources, just return module.name.resource_name
                return f"module.{module_name}.{resource_name}"
            # e.g., model.gpt4 -> gpt4, step.process -> process
            if parts[0] == expected_prefix:
                return ".".join(parts[1:])
            return ".".join(parts)
        elif isinstance(ref, str):
            return ref
        return None

    def _provider_ref_to_key(self, ref: Any) -> str:
        """Convert provider reference to provider key for ProvidersConfig.

        e.g., provider.llm.openai.default -> openai (or openai_default if not default)
        """
        if isinstance(ref, Reference):
            # provider.llm.openai.default -> ["provider", "llm", "openai", "default"]
            parts = ref.parts
            if len(parts) >= 3 and parts[0] == "provider" and parts[1] == "llm":
                vendor = parts[2]
                name = parts[3] if len(parts) > 3 else "default"
                if name == "default":
                    return vendor
                return f"{vendor}_{name}"
        return str(ref)

    def _nested_block_to_dict(self, block: NestedBlock) -> dict[str, Any]:
        """Convert nested block attributes to a dictionary."""
        result: dict[str, Any] = {}
        for attr in block.attributes:
            result[attr.name] = self._value_to_expr(attr.value)
        return result

    def _value_to_expr(self, value: Value) -> Any:
        """Convert AST value to expression string or primitive.

        For expression types (ConditionalExpr, ComparisonExpr, etc.), we convert
        to a string representation that can be evaluated at runtime. Alternatively,
        for static expressions (no state refs), we can evaluate at compile time.
        """
        if isinstance(value, Reference):
            # Convert to $-prefixed expression for runtime
            path = value.path
            if path.startswith("input.") or path.startswith("result.") or path.startswith("state."):
                return f"${path}"
            return path
        elif isinstance(value, StateRef):
            # State references are already $-prefixed
            return value.path
        elif isinstance(value, ConditionalExpr):
            # Check if static (can evaluate at compile time)
            if self._is_static_expr(value):
                return self._eval_static_conditional(value)
            # Convert to string representation for runtime evaluation
            return self._expr_to_string(value)
        elif isinstance(value, (ComparisonExpr, AndExpr, OrExpr, NotExpr)):
            # Check if static
            if self._is_static_expr(value):
                return self._eval_static_expr(value)
            # Convert to string for runtime
            return self._expr_to_string(value)
        elif isinstance(value, list):
            return [self._value_to_expr(v) for v in value]
        elif isinstance(value, VarRef):
            resolved, _ = self._resolve_variable(value)
            return resolved
        else:
            return value

    def _is_static_expr(self, expr: Any) -> bool:
        """Check if an expression contains only static values (no state refs)."""
        if isinstance(expr, StateRef):
            return False
        elif isinstance(expr, ConditionalExpr):
            return (
                self._is_static_expr(expr.condition)
                and self._is_static_expr(expr.true_value)
                and self._is_static_expr(expr.false_value)
            )
        elif isinstance(expr, ComparisonExpr):
            return self._is_static_expr(expr.left) and self._is_static_expr(expr.right)
        elif isinstance(expr, (AndExpr, OrExpr)):
            return all(self._is_static_expr(op) for op in expr.operands)
        elif isinstance(expr, NotExpr):
            return self._is_static_expr(expr.operand)
        elif isinstance(expr, (Reference, VarRef)):
            return True  # These don't depend on runtime state
        elif isinstance(expr, (str, int, float, bool)):
            return True
        return True

    def _eval_static_expr(self, expr: Any) -> Any:
        """Evaluate a static expression at compile time."""
        if isinstance(expr, ComparisonExpr):
            left = self._eval_static_expr(expr.left)
            right = self._eval_static_expr(expr.right)
            ops = {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<": lambda a, b: a < b,
                ">": lambda a, b: a > b,
                "<=": lambda a, b: a <= b,
                ">=": lambda a, b: a >= b,
            }
            op_func = ops.get(expr.operator)
            if op_func:
                return op_func(left, right)
        elif isinstance(expr, AndExpr):
            return all(self._eval_static_expr(op) for op in expr.operands)
        elif isinstance(expr, OrExpr):
            return any(self._eval_static_expr(op) for op in expr.operands)
        elif isinstance(expr, NotExpr):
            return not self._eval_static_expr(expr.operand)
        elif isinstance(expr, (str, int, float, bool)):
            return expr
        return expr

    def _eval_static_conditional(self, expr: ConditionalExpr) -> Any:
        """Evaluate a static conditional expression at compile time."""
        condition = self._eval_static_expr(expr.condition)
        if self._to_bool(condition):
            return self._value_to_expr(expr.true_value)
        else:
            return self._value_to_expr(expr.false_value)

    def _to_bool(self, value: Any) -> bool:
        """Convert a value to boolean for condition evaluation."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ("false", "no", "0", "")
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return bool(value)

    def _expr_to_string(self, expr: Any) -> str:
        """Convert an expression AST to a string representation for runtime."""
        if isinstance(expr, StateRef):
            return expr.path
        elif isinstance(expr, Reference):
            path = expr.path
            if path.startswith("input.") or path.startswith("state.") or path.startswith("result."):
                return f"${path}"
            return path
        elif isinstance(expr, ConditionalExpr):
            cond = self._expr_to_string(expr.condition)
            true_val = self._expr_to_string(expr.true_value)
            false_val = self._expr_to_string(expr.false_value)
            return f"{cond} ? {true_val} : {false_val}"
        elif isinstance(expr, ComparisonExpr):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"{left} {expr.operator} {right}"
        elif isinstance(expr, AndExpr):
            parts = [self._expr_to_string(op) for op in expr.operands]
            return " && ".join(parts)
        elif isinstance(expr, OrExpr):
            parts = [self._expr_to_string(op) for op in expr.operands]
            return " || ".join(parts)
        elif isinstance(expr, NotExpr):
            operand = self._expr_to_string(expr.operand)
            return f"!{operand}"
        elif isinstance(expr, str):
            # Quote strings for safety
            return f'"{expr}"'
        elif isinstance(expr, (int, float, bool)):
            return str(expr).lower() if isinstance(expr, bool) else str(expr)
        return str(expr)

    def _value_to_str(self, value: Value) -> str:
        """Convert value to string."""
        if isinstance(value, Reference):
            return value.path
        elif isinstance(value, VarRef):
            resolved, _ = self._resolve_variable(value)
            return resolved
        else:
            return str(value)


def normalize_agentform(
    agentform_file: AgentformFile,
    resolution: ResolutionResult,
    variables: dict[str, Any] | None = None,
    loaded_modules: dict[str, Any] | None = None,  # dict[str, LoadedModule]
) -> SpecRoot:
    """Normalize Agentform AST to SpecRoot.

    Args:
        agentform_file: Parsed Agentform AST
        resolution: Result from reference resolution
        variables: Dictionary of variable values to substitute
        loaded_modules: Dictionary of loaded module instances

    Returns:
        SpecRoot model compatible with existing pipeline

    Raises:
        NormalizationError: If normalization fails
    """
    normalizer = AgentformNormalizer(agentform_file, resolution, variables, loaded_modules)
    return normalizer.normalize()
