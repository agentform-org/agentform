"""Validation for Agentform specifications."""

from dataclasses import dataclass, field

from agentform_compiler.credentials import get_env_var_name, is_env_reference
from agentform_schema.models import SpecRoot, StepType


@dataclass
class ValidationError:
    """A single validation error."""

    path: str
    message: str


@dataclass
class ValidationResult:
    """Result of validation."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def add_error(self, path: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(path, message))

    def add_warning(self, path: str, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(path, message))


def validate_spec(spec: SpecRoot, check_env: bool = True) -> ValidationResult:
    """Validate an Agentform specification.

    This performs semantic validation beyond schema validation:
    - References between components are valid
    - Environment variables exist (if check_env=True)
    - Workflow steps are correctly configured

    Args:
        spec: The parsed specification
        check_env: Whether to check env vars exist

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult()

    # Build lookup maps
    server_names = {s.name for s in spec.servers}
    capability_names = {c.name for c in spec.capabilities}
    policy_names = {p.name for p in spec.policies}
    agent_names = {a.name for a in spec.agents}
    provider_names = set(spec.providers.llm.keys())

    # Collect env var references
    env_refs: list[tuple[str, str]] = []  # (path, reference)

    # Validate providers
    for name, provider in spec.providers.llm.items():
        path = f"providers.llm.{name}"
        if is_env_reference(provider.api_key):
            env_refs.append((f"{path}.api_key", provider.api_key))
        elif not provider.api_key:
            # Empty API key is an error; direct values are allowed (from variable substitution)
            result.add_error(
                f"{path}.api_key",
                "API key is required",
            )

    # Validate servers
    for i, server in enumerate(spec.servers):
        path = f"servers[{i}]"
        if server.auth and server.auth.token and is_env_reference(server.auth.token):
            env_refs.append((f"{path}.auth.token", server.auth.token))

    # Validate capabilities
    for i, cap in enumerate(spec.capabilities):
        path = f"capabilities[{i}]"
        if cap.server not in server_names:
            result.add_error(
                f"{path}.server",
                f"Server '{cap.server}' not found",
            )

    # Validate agents
    for i, agent in enumerate(spec.agents):
        path = f"agents[{i}]"

        # Check provider exists
        if agent.provider not in provider_names:
            result.add_error(
                f"{path}.provider",
                f"Provider '{agent.provider}' not found in providers.llm",
            )

        # Check policy exists (if specified)
        if agent.policy and agent.policy not in policy_names:
            result.add_error(
                f"{path}.policy",
                f"Policy '{agent.policy}' not found",
            )

        # Check capabilities exist
        for j, cap_name in enumerate(agent.allow):
            if cap_name not in capability_names:
                result.add_error(
                    f"{path}.allow[{j}]",
                    f"Capability '{cap_name}' not found",
                )

    # Validate workflows
    for i, workflow in enumerate(spec.workflows):
        path = f"workflows[{i}]"
        step_ids = {s.id for s in workflow.steps}

        # Check entry step exists
        if workflow.entry not in step_ids:
            result.add_error(
                f"{path}.entry",
                f"Entry step '{workflow.entry}' not found",
            )

        # Validate each step
        for j, step in enumerate(workflow.steps):
            step_path = f"{path}.steps[{j}]"

            # Check next step exists
            if step.next and step.next not in step_ids and step.next != "end":
                result.add_error(
                    f"{step_path}.next",
                    f"Step '{step.next}' not found",
                )

            # Type-specific validation
            if step.type == StepType.LLM:
                if not step.agent:
                    result.add_error(f"{step_path}.agent", "LLM step requires 'agent'")
                elif step.agent not in agent_names:
                    result.add_error(
                        f"{step_path}.agent",
                        f"Agent '{step.agent}' not found",
                    )

            elif step.type == StepType.CALL:
                if not step.capability:
                    result.add_error(
                        f"{step_path}.capability",
                        "Call step requires 'capability'",
                    )
                elif step.capability not in capability_names:
                    result.add_error(
                        f"{step_path}.capability",
                        f"Capability '{step.capability}' not found",
                    )

            elif step.type == StepType.CONDITION:
                if not step.condition:
                    result.add_error(
                        f"{step_path}.condition",
                        "Condition step requires 'condition'",
                    )
                if step.on_true and step.on_true not in step_ids:
                    result.add_error(
                        f"{step_path}.on_true",
                        f"Step '{step.on_true}' not found",
                    )
                if step.on_false and step.on_false not in step_ids:
                    result.add_error(
                        f"{step_path}.on_false",
                        f"Step '{step.on_false}' not found",
                    )

            elif step.type == StepType.HUMAN_APPROVAL:
                if step.on_approve and step.on_approve not in step_ids:
                    result.add_error(
                        f"{step_path}.on_approve",
                        f"Step '{step.on_approve}' not found",
                    )
                if step.on_reject and step.on_reject not in step_ids:
                    result.add_error(
                        f"{step_path}.on_reject",
                        f"Step '{step.on_reject}' not found",
                    )

    # Check environment variables exist
    if check_env:
        import os

        for path, ref in env_refs:
            var_name = get_env_var_name(ref)
            if var_name and var_name not in os.environ:
                result.add_warning(
                    path,
                    f"Environment variable '{var_name}' is not set",
                )

    return result
