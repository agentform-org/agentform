"""IR (Intermediate Representation) generation from validated specs."""

from agentform_compiler.credentials import get_env_var_name, resolve_env_var
from agentform_schema.ir import (
    CompiledSpec,
    ResolvedAgent,
    ResolvedCapability,
    ResolvedCredential,
    ResolvedPolicy,
    ResolvedProvider,
    ResolvedServer,
    ResolvedStep,
    ResolvedWorkflow,
)
from agentform_schema.models import (
    BudgetConfig,
    LLMProviderParams,
    SpecRoot,
)


class IRGenerationError(Exception):
    """Error during IR generation."""

    pass


def generate_ir(spec: SpecRoot, resolve_credentials: bool = True) -> CompiledSpec:
    """Generate IR from a validated specification.

    Args:
        spec: Validated specification
        resolve_credentials: Whether to resolve env vars to actual values

    Returns:
        Compiled specification (IR)

    Raises:
        IRGenerationError: If IR generation fails
    """
    # Resolve providers
    providers: dict[str, ResolvedProvider] = {}
    for name, provider in spec.providers.llm.items():
        var_name = get_env_var_name(provider.api_key)

        if var_name:
            # env:VAR_NAME format - resolve from environment
            value = None
            if resolve_credentials:
                value = resolve_env_var(provider.api_key, required=False)
            api_key = ResolvedCredential(env_var=var_name, value=value)
        else:
            # Direct value (from variable substitution)
            api_key = ResolvedCredential(env_var="DIRECT_VALUE", value=provider.api_key)

        providers[name] = ResolvedProvider(
            name=name,
            api_key=api_key,
            default_params=provider.default_params or LLMProviderParams(),
        )

    # Resolve servers
    servers: dict[str, ResolvedServer] = {}
    for server in spec.servers:
        auth_token = None
        if server.auth and server.auth.token:
            var_name = get_env_var_name(server.auth.token)
            if var_name:
                # env:VAR_NAME format
                value = None
                if resolve_credentials:
                    value = resolve_env_var(server.auth.token, required=False)
                auth_token = ResolvedCredential(env_var=var_name, value=value)
            else:
                # Direct value (from variable substitution)
                auth_token = ResolvedCredential(env_var="DIRECT_VALUE", value=server.auth.token)

        servers[server.name] = ResolvedServer(
            name=server.name,
            command=server.command,
            auth_token=auth_token,
        )

    # Resolve capabilities (method schemas will be populated by MCP discovery)
    capabilities: dict[str, ResolvedCapability] = {}
    for cap in spec.capabilities:
        capabilities[cap.name] = ResolvedCapability(
            name=cap.name,
            server_name=cap.server,
            method_name=cap.method,
            method_schema=None,  # Populated during MCP resolution
            side_effect=cap.side_effect,
            requires_approval=cap.requires_approval,
        )

    # Resolve policies
    policies: dict[str, ResolvedPolicy] = {}
    for policy in spec.policies:
        policies[policy.name] = ResolvedPolicy(
            name=policy.name,
            budgets=policy.budgets or BudgetConfig(),
        )

    # Resolve agents
    agents: dict[str, ResolvedAgent] = {}
    for agent in spec.agents:
        # Merge provider defaults with agent params
        resolved_provider = providers.get(agent.provider)
        if resolved_provider is None:
            raise IRGenerationError(
                f"Provider '{agent.provider}' not found for agent '{agent.name}'"
            )

        merged_params = LLMProviderParams()
        if resolved_provider.default_params:
            merged_params = resolved_provider.default_params.model_copy()
        if agent.params:
            # Override with agent-specific params
            for field_name in type(agent.params).model_fields:
                agent_value = getattr(agent.params, field_name)
                if agent_value is not None:
                    setattr(merged_params, field_name, agent_value)

        agents[agent.name] = ResolvedAgent(
            name=agent.name,
            provider_name=agent.provider,
            model_preference=agent.model.preference,
            model_fallback=agent.model.fallback,
            params=merged_params,
            instructions=agent.instructions,
            allowed_capabilities=agent.allow,
            policy_name=agent.policy,
        )

    # Resolve workflows
    workflows: dict[str, ResolvedWorkflow] = {}
    for workflow in spec.workflows:
        steps: dict[str, ResolvedStep] = {}
        for step in workflow.steps:
            resolved_step = ResolvedStep(
                id=step.id,
                type=step.type,
                agent_name=step.agent,
                input_mapping=step.input,
                capability_name=step.capability,
                args_mapping=step.args,
                condition_expr=step.condition,
                on_true_step=step.on_true,
                on_false_step=step.on_false,
                payload_expr=step.payload,
                on_approve_step=step.on_approve,
                on_reject_step=step.on_reject,
                save_as=step.save_as,
                next_step=step.next,
            )
            steps[step.id] = resolved_step

        workflows[workflow.name] = ResolvedWorkflow(
            name=workflow.name,
            entry_step=workflow.entry,
            steps=steps,
        )

    return CompiledSpec(
        version=spec.version,
        project_name=spec.project.name,
        providers=providers,
        servers=servers,
        capabilities=capabilities,
        policies=policies,
        agents=agents,
        workflows=workflows,
    )
