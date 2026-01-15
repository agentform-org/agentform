"""Validator for ACP native schema.

Validates ACP-specific rules beyond reference resolution,
such as required fields, valid values, and step type requirements.
"""

from dataclasses import dataclass, field

from acp_compiler.acp_ast import (
    ACPFile,
    AgentBlock,
    CapabilityBlock,
    EnvCall,
    ModelBlock,
    PolicyBlock,
    ProviderBlock,
    ServerBlock,
    SourceLocation,
    StepBlock,
    WorkflowBlock,
)
from acp_compiler.acp_resolver import ResolutionResult


@dataclass
class ACPValidationError:
    """A validation error."""

    path: str
    message: str
    location: SourceLocation | None = None

    def __str__(self) -> str:
        if self.location:
            return f"{self.location}: {self.path}: {self.message}"
        return f"{self.path}: {self.message}"


@dataclass
class ACPValidationResult:
    """Result of ACP validation."""

    errors: list[ACPValidationError] = field(default_factory=list)
    warnings: list[ACPValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def add_error(
        self,
        path: str,
        message: str,
        location: SourceLocation | None = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(ACPValidationError(path, message, location))

    def add_warning(
        self,
        path: str,
        message: str,
        location: SourceLocation | None = None,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ACPValidationError(path, message, location))


# Valid step types
VALID_STEP_TYPES = {"llm", "call", "condition", "human_approval", "end", "router", "tool"}

# Valid side effects
VALID_SIDE_EFFECTS = {"read", "write"}

# Valid transports
VALID_TRANSPORTS = {"stdio"}


class ACPValidator:
    """Validates an ACP AST for correctness."""

    def __init__(
        self,
        acp_file: ACPFile,
        resolution: ResolutionResult,
        check_env: bool = True,
    ):
        self.acp_file = acp_file
        self.resolution = resolution
        self.check_env = check_env
        self.result = ACPValidationResult()

    def validate(self) -> ACPValidationResult:
        """Run all validations.

        Returns:
            ACPValidationResult with errors and warnings
        """
        # Check acp block
        self._validate_acp_block()

        # Check providers
        for provider in self.acp_file.providers:
            self._validate_provider(provider)

        # Check servers
        for server in self.acp_file.servers:
            self._validate_server(server)

        # Check capabilities
        for capability in self.acp_file.capabilities:
            self._validate_capability(capability)

        # Check policies
        for policy in self.acp_file.policies:
            self._validate_policy(policy)

        # Check models
        for model in self.acp_file.models:
            self._validate_model(model)

        # Check agents
        for agent in self.acp_file.agents:
            self._validate_agent(agent)

        # Check workflows
        for workflow in self.acp_file.workflows:
            self._validate_workflow(workflow)

        # Check environment variables if requested
        if self.check_env:
            self._validate_env_vars()

        return self.result

    def _validate_acp_block(self) -> None:
        """Validate the acp metadata block."""
        if self.acp_file.acp is None:
            self.result.add_error("acp", "Missing required 'acp' block")
            return

        acp = self.acp_file.acp

        if not acp.version:
            self.result.add_error("acp.version", "Missing required 'version' field", acp.location)

        if not acp.project:
            self.result.add_error("acp.project", "Missing required 'project' field", acp.location)

    def _validate_provider(self, provider: ProviderBlock) -> None:
        """Validate a provider block."""
        path = f"provider.{provider.full_name}"

        # Check api_key is present and is an env call
        api_key = provider.get_attribute("api_key")
        if api_key is None:
            self.result.add_error(
                f"{path}.api_key",
                "Missing required 'api_key' field",
                provider.location,
            )
        elif not isinstance(api_key, EnvCall):
            self.result.add_error(
                f"{path}.api_key",
                "api_key must use env() function",
                provider.location,
            )

    def _validate_server(self, server: ServerBlock) -> None:
        """Validate a server block."""
        path = f"server.{server.name}"

        # Check command is present
        command = server.get_attribute("command")
        if command is None:
            self.result.add_error(
                f"{path}.command",
                "Missing required 'command' field",
                server.location,
            )
        elif not isinstance(command, list):
            self.result.add_error(
                f"{path}.command",
                "command must be an array",
                server.location,
            )

        # Check transport if present
        transport = server.get_attribute("transport")
        if transport is not None and transport not in VALID_TRANSPORTS:
            self.result.add_error(
                f"{path}.transport",
                f"Invalid transport: {transport}. Valid values: {', '.join(VALID_TRANSPORTS)}",
                server.location,
            )

    def _validate_capability(self, capability: CapabilityBlock) -> None:
        """Validate a capability block."""
        path = f"capability.{capability.name}"

        # Check server reference
        server = capability.get_attribute("server")
        if server is None:
            self.result.add_error(
                f"{path}.server",
                "Missing required 'server' field",
                capability.location,
            )

        # Check method
        method = capability.get_attribute("method")
        if method is None:
            self.result.add_error(
                f"{path}.method",
                "Missing required 'method' field",
                capability.location,
            )

        # Check side_effect if present
        side_effect = capability.get_attribute("side_effect")
        if side_effect is not None and side_effect not in VALID_SIDE_EFFECTS:
            self.result.add_error(
                f"{path}.side_effect",
                f"Invalid side_effect: {side_effect}. Valid values: {', '.join(VALID_SIDE_EFFECTS)}",
                capability.location,
            )

    def _validate_policy(self, policy: PolicyBlock) -> None:
        """Validate a policy block."""
        path = f"policy.{policy.name}"

        # Budgets blocks are optional but if present should have valid fields
        for budget_block in policy.get_budgets_blocks():
            for attr in budget_block.attributes:
                if attr.name not in (
                    "max_cost_usd_per_run",
                    "max_capability_calls",
                    "timeout_seconds",
                ):
                    self.result.add_warning(
                        f"{path}.budgets.{attr.name}",
                        f"Unknown budget field: {attr.name}",
                        attr.location,
                    )

    def _validate_model(self, model: ModelBlock) -> None:
        """Validate a model block."""
        path = f"model.{model.name}"

        # Check provider reference
        provider = model.get_attribute("provider")
        if provider is None:
            self.result.add_error(
                f"{path}.provider",
                "Missing required 'provider' field",
                model.location,
            )

        # Check id (model identifier)
        model_id = model.get_attribute("id")
        if model_id is None:
            self.result.add_error(
                f"{path}.id",
                "Missing required 'id' field (model identifier)",
                model.location,
            )

    def _validate_agent(self, agent: AgentBlock) -> None:
        """Validate an agent block."""
        path = f"agent.{agent.name}"

        # Check model reference (required)
        model = agent.get_attribute("model")
        if model is None:
            self.result.add_error(
                f"{path}.model",
                "Missing required 'model' field",
                agent.location,
            )

        # Check instructions (required)
        instructions = agent.get_attribute("instructions")
        if instructions is None:
            self.result.add_error(
                f"{path}.instructions",
                "Missing required 'instructions' field",
                agent.location,
            )

        # Validate fallback_models is an array if present
        fallback = agent.get_attribute("fallback_models")
        if fallback is not None and not isinstance(fallback, list):
            self.result.add_error(
                f"{path}.fallback_models",
                "fallback_models must be an array",
                agent.location,
            )

        # Validate allow is an array if present
        allow = agent.get_attribute("allow")
        if allow is not None and not isinstance(allow, list):
            self.result.add_error(
                f"{path}.allow",
                "allow must be an array",
                agent.location,
            )

    def _validate_workflow(self, workflow: WorkflowBlock) -> None:
        """Validate a workflow block."""
        path = f"workflow.{workflow.name}"

        # Check entry step
        entry = workflow.get_attribute("entry")
        if entry is None:
            self.result.add_error(
                f"{path}.entry",
                "Missing required 'entry' field",
                workflow.location,
            )

        # Check at least one step
        if not workflow.steps:
            self.result.add_error(
                f"{path}.steps",
                "Workflow must have at least one step",
                workflow.location,
            )

        # Validate each step
        for step in workflow.steps:
            self._validate_step(step, workflow.name)

    def _validate_step(self, step: StepBlock, workflow_name: str) -> None:
        """Validate a workflow step."""
        path = f"workflow.{workflow_name}.step.{step.step_id}"

        # Check type
        step_type = step.get_attribute("type")
        if step_type is None:
            self.result.add_error(
                f"{path}.type",
                "Missing required 'type' field",
                step.location,
            )
            return

        if step_type not in VALID_STEP_TYPES:
            self.result.add_error(
                f"{path}.type",
                f"Invalid step type: {step_type}. Valid types: {', '.join(sorted(VALID_STEP_TYPES))}",
                step.location,
            )
            return

        # Type-specific validation
        if step_type == "llm":
            self._validate_llm_step(step, path)
        elif step_type == "call" or step_type == "tool":
            self._validate_call_step(step, path)
        elif step_type == "condition" or step_type == "router":
            self._validate_condition_step(step, path)
        elif step_type == "human_approval":
            self._validate_human_approval_step(step, path)
        # "end" type needs no additional validation

    def _validate_llm_step(self, step: StepBlock, path: str) -> None:
        """Validate an LLM step."""
        # Require agent
        agent = step.get_attribute("agent")
        if agent is None:
            self.result.add_error(
                f"{path}.agent",
                "LLM step requires 'agent' field",
                step.location,
            )

    def _validate_call_step(self, step: StepBlock, path: str) -> None:
        """Validate a call/tool step."""
        # Require capability
        capability = step.get_attribute("capability")
        if capability is None:
            self.result.add_error(
                f"{path}.capability",
                "Call step requires 'capability' field",
                step.location,
            )

    def _validate_condition_step(self, step: StepBlock, path: str) -> None:
        """Validate a condition/router step."""
        # Require condition expression
        condition = step.get_attribute("condition")
        if condition is None:
            self.result.add_error(
                f"{path}.condition",
                "Condition step requires 'condition' field",
                step.location,
            )

        # Should have on_true and on_false
        on_true = step.get_attribute("on_true")
        on_false = step.get_attribute("on_false")
        if on_true is None and on_false is None:
            self.result.add_warning(
                path,
                "Condition step should have 'on_true' and/or 'on_false' fields",
                step.location,
            )

    def _validate_human_approval_step(self, step: StepBlock, path: str) -> None:
        """Validate a human_approval step."""
        # Should have on_approve and on_reject
        on_approve = step.get_attribute("on_approve")
        on_reject = step.get_attribute("on_reject")
        if on_approve is None and on_reject is None:
            self.result.add_warning(
                path,
                "Human approval step should have 'on_approve' and/or 'on_reject' fields",
                step.location,
            )

    def _validate_env_vars(self) -> None:
        """Check that referenced environment variables are set."""
        import os

        env_refs = self._collect_env_refs()
        for path, var_name in env_refs:
            if var_name not in os.environ:
                self.result.add_warning(
                    path,
                    f"Environment variable '{var_name}' is not set",
                )

    def _collect_env_refs(self) -> list[tuple[str, str]]:
        """Collect all env() references from the AST."""
        refs: list[tuple[str, str]] = []

        # From providers
        for provider in self.acp_file.providers:
            path = f"provider.{provider.full_name}"
            api_key = provider.get_attribute("api_key")
            if isinstance(api_key, EnvCall):
                refs.append((f"{path}.api_key", api_key.var_name))

        # From servers (auth tokens)
        for server in self.acp_file.servers:
            path = f"server.{server.name}"
            for block in server.blocks:
                if block.block_type == "auth":
                    token = block.get_attribute("token")
                    if isinstance(token, EnvCall):
                        refs.append((f"{path}.auth.token", token.var_name))

        return refs


def validate_acp(
    acp_file: ACPFile,
    resolution: ResolutionResult,
    check_env: bool = True,
) -> ACPValidationResult:
    """Validate an ACP AST.

    Args:
        acp_file: Parsed ACP AST
        resolution: Result from reference resolution
        check_env: Whether to check environment variables exist

    Returns:
        ACPValidationResult with errors and warnings
    """
    validator = ACPValidator(acp_file, resolution, check_env)
    return validator.validate()

