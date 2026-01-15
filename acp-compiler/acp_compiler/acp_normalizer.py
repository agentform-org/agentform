"""Normalizer that transforms ACP AST to existing SpecRoot format.

This module bridges the new ACP native schema with the existing
YAML-based infrastructure by converting the parsed AST into
the same SpecRoot model used by YAML parsing.
"""

from typing import Any

from acp_compiler.acp_ast import (
    ACPFile,
    AgentBlock,
    CapabilityBlock,
    EnvCall,
    ModelBlock,
    NestedBlock,
    PolicyBlock,
    ProviderBlock,
    Reference,
    ServerBlock,
    SourceLocation,
    StepBlock,
    Value,
    WorkflowBlock,
)
from acp_compiler.acp_resolver import ResolutionResult
from acp_schema.models import (
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


class ACPNormalizer:
    """Transforms ACP AST to SpecRoot."""

    def __init__(self, acp_file: ACPFile, resolution: ResolutionResult):
        self.acp_file = acp_file
        self.resolution = resolution
        # Cache for resolved model info
        self._model_cache: dict[str, tuple[str, str, LLMProviderParams | None]] = {}

    def normalize(self) -> SpecRoot:
        """Normalize ACP AST to SpecRoot.

        Returns:
            SpecRoot model compatible with existing pipeline

        Raises:
            NormalizationError: If normalization fails
        """
        # Build model cache first (maps model name -> (provider, model_id, params))
        self._build_model_cache()

        return SpecRoot(
            version=self._get_version(),
            project=self._normalize_project(),
            providers=self._normalize_providers(),
            servers=self._normalize_servers(),
            capabilities=self._normalize_capabilities(),
            policies=self._normalize_policies(),
            agents=self._normalize_agents(),
            workflows=self._normalize_workflows(),
        )

    def _get_version(self) -> str:
        """Get version from acp block."""
        if self.acp_file.acp and self.acp_file.acp.version:
            return self.acp_file.acp.version
        return "0.1"

    def _normalize_project(self) -> ProjectConfig:
        """Normalize project configuration."""
        name = "unnamed"
        if self.acp_file.acp and self.acp_file.acp.project:
            name = self.acp_file.acp.project
        return ProjectConfig(name=name)

    def _normalize_providers(self) -> ProvidersConfig:
        """Normalize provider blocks to ProvidersConfig."""
        llm_providers: dict[str, LLMProviderConfig] = {}

        for provider in self.acp_file.providers:
            # Parse provider type (e.g., "llm.openai" -> category="llm", vendor="openai")
            parts = provider.provider_type.split(".")
            if len(parts) >= 2 and parts[0] == "llm":
                vendor = ".".join(parts[1:])  # Handle nested vendor names
                # Create unique key: vendor_name or just vendor if name is "default"
                key = f"{vendor}" if provider.name == "default" else f"{vendor}_{provider.name}"

                # Get api_key
                api_key_val = provider.get_attribute("api_key")
                if isinstance(api_key_val, EnvCall):
                    api_key = f"env:{api_key_val.var_name}"
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

        for server in self.acp_file.servers:
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
                    if isinstance(token, EnvCall):
                        auth = ServerAuthConfig(token=f"env:{token.var_name}")
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

        for cap in self.acp_file.capabilities:
            # Get server reference
            server_ref = cap.get_attribute("server")
            server_name = self._ref_to_name(server_ref, "server")

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

        for policy in self.acp_file.policies:
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
        for model in self.acp_file.models:
            # Get provider reference
            provider_ref = model.get_attribute("provider")
            provider_name = self._provider_ref_to_key(provider_ref)

            # Get model id
            model_id = model.get_attribute("id")
            if not isinstance(model_id, str):
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

        for agent in self.acp_file.agents:
            # Get primary model reference
            model_ref = agent.get_attribute("model")
            model_name = self._ref_to_name(model_ref, "model")
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

        for workflow in self.acp_file.workflows:
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
        condition = step.get_attribute("condition")
        if not isinstance(condition, str):
            condition = None

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
        """Extract name from a reference, removing the type prefix."""
        if isinstance(ref, Reference):
            # e.g., model.gpt4 -> gpt4, step.process -> process
            if ref.parts[0] == expected_prefix:
                return ".".join(ref.parts[1:])
            return ".".join(ref.parts)
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
        """Convert AST value to expression string or primitive."""
        if isinstance(value, Reference):
            # Convert to $-prefixed expression for runtime
            path = value.path
            if path.startswith("input."):
                return f"${path}"
            elif path.startswith("result."):
                return f"${path}"
            elif path.startswith("state."):
                return f"${path}"
            return path
        elif isinstance(value, list):
            return [self._value_to_expr(v) for v in value]
        elif isinstance(value, EnvCall):
            return f"env:{value.var_name}"
        else:
            return value

    def _value_to_str(self, value: Value) -> str:
        """Convert value to string."""
        if isinstance(value, Reference):
            return value.path
        elif isinstance(value, EnvCall):
            return f"env:{value.var_name}"
        else:
            return str(value)


def normalize_acp(acp_file: ACPFile, resolution: ResolutionResult) -> SpecRoot:
    """Normalize ACP AST to SpecRoot.

    Args:
        acp_file: Parsed ACP AST
        resolution: Result from reference resolution

    Returns:
        SpecRoot model compatible with existing pipeline

    Raises:
        NormalizationError: If normalization fails
    """
    normalizer = ACPNormalizer(acp_file, resolution)
    return normalizer.normalize()

