"""Policy enforcement for workflow execution."""

import time
from dataclasses import dataclass, field

from agentform_schema.ir import ResolvedPolicy


@dataclass
class PolicyContext:
    """Tracks policy metrics during execution."""

    start_time: float = field(default_factory=time.time)
    capability_calls: int = 0
    total_cost_usd: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)

    def add_capability_call(self) -> None:
        """Record a capability call."""
        self.capability_calls += 1

    def add_cost(self, cost_usd: float) -> None:
        """Record cost."""
        self.total_cost_usd += cost_usd

    def add_tokens(self, model: str, tokens: int) -> None:
        """Record token usage."""
        if model not in self.token_usage:
            self.token_usage[model] = 0
        self.token_usage[model] += tokens

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time


class PolicyViolation(Exception):
    """Raised when a policy constraint is violated."""

    def __init__(self, policy_name: str, constraint: str, message: str):
        self.policy_name = policy_name
        self.constraint = constraint
        super().__init__(f"Policy '{policy_name}' violation ({constraint}): {message}")


class PolicyEnforcer:
    """Enforces policies during workflow execution."""

    def __init__(self, policies: dict[str, ResolvedPolicy]):
        """Initialize enforcer.

        Args:
            policies: Available policies by name
        """
        self._policies = policies
        self._contexts: dict[str, PolicyContext] = {}

    def start_context(self, context_id: str) -> PolicyContext:
        """Start a new policy context (e.g., for a workflow run).

        Args:
            context_id: Unique identifier for the context

        Returns:
            New policy context
        """
        context = PolicyContext()
        self._contexts[context_id] = context
        return context

    def get_context(self, context_id: str) -> PolicyContext | None:
        """Get an existing context."""
        return self._contexts.get(context_id)

    def end_context(self, context_id: str) -> None:
        """End and remove a context."""
        self._contexts.pop(context_id, None)

    def check_before_capability_call(
        self,
        context_id: str,
        policy_name: str | None,
    ) -> None:
        """Check policy before a capability call.

        Args:
            context_id: Context identifier
            policy_name: Policy to check (None = no policy)

        Raises:
            PolicyViolation: If call would violate policy
        """
        if not policy_name:
            return

        policy = self._policies.get(policy_name)
        if not policy:
            return

        context = self._contexts.get(context_id)
        if not context:
            return

        budgets = policy.budgets

        # Check capability call limit
        if (
            budgets.max_capability_calls is not None
            and context.capability_calls >= budgets.max_capability_calls
        ):
            raise PolicyViolation(
                policy_name,
                "max_capability_calls",
                f"Limit of {budgets.max_capability_calls} calls reached",
            )

        # Check timeout
        if (
            budgets.timeout_seconds is not None
            and context.elapsed_seconds >= budgets.timeout_seconds
        ):
            raise PolicyViolation(
                policy_name,
                "timeout_seconds",
                f"Timeout of {budgets.timeout_seconds}s exceeded",
            )

    def record_capability_call(self, context_id: str) -> None:
        """Record a capability call.

        Args:
            context_id: Context identifier
        """
        context = self._contexts.get(context_id)
        if context:
            context.add_capability_call()

    def check_cost(
        self,
        context_id: str,
        policy_name: str | None,
        cost_usd: float,
    ) -> None:
        """Check and record cost.

        Args:
            context_id: Context identifier
            policy_name: Policy to check
            cost_usd: Cost to add

        Raises:
            PolicyViolation: If cost would exceed budget
        """
        context = self._contexts.get(context_id)
        if not context:
            return

        context.add_cost(cost_usd)

        if not policy_name:
            return

        policy = self._policies.get(policy_name)
        if not policy:
            return

        budgets = policy.budgets
        if (
            budgets.max_cost_usd_per_run is not None
            and context.total_cost_usd > budgets.max_cost_usd_per_run
        ):
            raise PolicyViolation(
                policy_name,
                "max_cost_usd_per_run",
                f"Cost ${context.total_cost_usd:.4f} exceeds budget ${budgets.max_cost_usd_per_run:.2f}",
            )

    def check_timeout(self, context_id: str, policy_name: str | None) -> None:
        """Check if timeout has been exceeded.

        Args:
            context_id: Context identifier
            policy_name: Policy to check

        Raises:
            PolicyViolation: If timeout exceeded
        """
        if not policy_name:
            return

        policy = self._policies.get(policy_name)
        if not policy:
            return

        context = self._contexts.get(context_id)
        if not context:
            return

        budgets = policy.budgets
        if (
            budgets.timeout_seconds is not None
            and context.elapsed_seconds >= budgets.timeout_seconds
        ):
            raise PolicyViolation(
                policy_name,
                "timeout_seconds",
                f"Timeout of {budgets.timeout_seconds}s exceeded",
            )
