"""Tests for policy enforcement."""

import time

import pytest

from agentform_runtime.policy import (
    PolicyContext,
    PolicyEnforcer,
    PolicyViolation,
)
from agentform_schema.ir import ResolvedPolicy
from agentform_schema.models import BudgetConfig


class TestPolicyContext:
    """Tests for PolicyContext class."""

    def test_init_defaults(self):
        """Test context initialization with defaults."""
        context = PolicyContext()
        assert context.capability_calls == 0
        assert context.total_cost_usd == 0.0
        assert context.token_usage == {}
        assert context.start_time > 0

    def test_add_capability_call(self):
        """Test recording capability calls."""
        context = PolicyContext()
        assert context.capability_calls == 0

        context.add_capability_call()
        assert context.capability_calls == 1

        context.add_capability_call()
        context.add_capability_call()
        assert context.capability_calls == 3

    def test_add_cost(self):
        """Test recording costs."""
        context = PolicyContext()

        context.add_cost(0.01)
        assert context.total_cost_usd == 0.01

        context.add_cost(0.05)
        assert context.total_cost_usd == pytest.approx(0.06)

    def test_add_tokens(self):
        """Test recording token usage."""
        context = PolicyContext()

        context.add_tokens("gpt-4", 100)
        assert context.token_usage == {"gpt-4": 100}

        context.add_tokens("gpt-4", 50)
        assert context.token_usage == {"gpt-4": 150}

        context.add_tokens("gpt-3.5-turbo", 200)
        assert context.token_usage == {"gpt-4": 150, "gpt-3.5-turbo": 200}

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        context = PolicyContext()
        initial = context.elapsed_seconds

        time.sleep(0.1)
        elapsed = context.elapsed_seconds

        assert elapsed >= 0.1
        assert elapsed > initial


class TestPolicyViolation:
    """Tests for PolicyViolation exception."""

    def test_violation_message(self):
        """Test violation message format."""
        exc = PolicyViolation("default", "timeout_seconds", "Timed out")

        assert exc.policy_name == "default"
        assert exc.constraint == "timeout_seconds"
        assert "default" in str(exc)
        assert "timeout_seconds" in str(exc)
        assert "Timed out" in str(exc)


class TestPolicyEnforcer:
    """Tests for PolicyEnforcer class."""

    @pytest.fixture
    def policies(self):
        """Create test policies."""
        return {
            "strict": ResolvedPolicy(
                name="strict",
                budgets=BudgetConfig(
                    max_cost_usd_per_run=1.00,
                    max_capability_calls=10,
                    timeout_seconds=60,
                ),
            ),
            "unlimited": ResolvedPolicy(
                name="unlimited",
                budgets=BudgetConfig(),
            ),
        }

    def test_init(self, policies):
        """Test enforcer initialization."""
        enforcer = PolicyEnforcer(policies)
        assert enforcer._policies == policies
        assert enforcer._contexts == {}

    def test_start_context(self, policies):
        """Test starting a policy context."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")

        assert isinstance(context, PolicyContext)
        assert "ctx-1" in enforcer._contexts

    def test_get_context(self, policies):
        """Test getting existing context."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        context = enforcer.get_context("ctx-1")
        assert context is not None

        missing = enforcer.get_context("missing")
        assert missing is None

    def test_end_context(self, policies):
        """Test ending a context."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        enforcer.end_context("ctx-1")
        assert enforcer.get_context("ctx-1") is None

        # Should not raise for missing context
        enforcer.end_context("missing")

    def test_check_before_capability_call_no_policy(self, policies):
        """Test check passes when no policy specified."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        # Should not raise
        enforcer.check_before_capability_call("ctx-1", None)

    def test_check_before_capability_call_within_limit(self, policies):
        """Test check passes within limit."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")
        context.capability_calls = 5  # Under limit of 10

        # Should not raise
        enforcer.check_before_capability_call("ctx-1", "strict")

    def test_check_before_capability_call_exceeds_limit(self, policies):
        """Test check raises when exceeding limit."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")
        context.capability_calls = 10  # At limit

        with pytest.raises(PolicyViolation) as exc_info:
            enforcer.check_before_capability_call("ctx-1", "strict")

        assert exc_info.value.constraint == "max_capability_calls"
        assert "10" in str(exc_info.value)

    def test_check_before_capability_call_unlimited(self, policies):
        """Test check passes for unlimited policy."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")
        context.capability_calls = 1000

        # Should not raise
        enforcer.check_before_capability_call("ctx-1", "unlimited")

    def test_check_before_capability_call_unknown_policy(self, policies):
        """Test check passes for unknown policy."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        # Should not raise for unknown policy
        enforcer.check_before_capability_call("ctx-1", "nonexistent")

    def test_record_capability_call(self, policies):
        """Test recording capability calls."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")

        assert context.capability_calls == 0
        enforcer.record_capability_call("ctx-1")
        assert context.capability_calls == 1
        enforcer.record_capability_call("ctx-1")
        assert context.capability_calls == 2

    def test_record_capability_call_missing_context(self, policies):
        """Test recording call with missing context does nothing."""
        enforcer = PolicyEnforcer(policies)
        # Should not raise
        enforcer.record_capability_call("missing")

    def test_check_cost_within_budget(self, policies):
        """Test check cost passes within budget."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        # Should not raise
        enforcer.check_cost("ctx-1", "strict", 0.50)

    def test_check_cost_exceeds_budget(self, policies):
        """Test check cost raises when exceeding budget."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        enforcer.check_cost("ctx-1", "strict", 0.60)  # First call OK

        with pytest.raises(PolicyViolation) as exc_info:
            enforcer.check_cost("ctx-1", "strict", 0.50)  # Total 1.10 > 1.00

        assert exc_info.value.constraint == "max_cost_usd_per_run"

    def test_check_cost_no_policy(self, policies):
        """Test check cost passes with no policy."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        # Should not raise and should still record cost
        enforcer.check_cost("ctx-1", None, 100.00)
        context = enforcer.get_context("ctx-1")
        assert context is not None
        assert context.total_cost_usd == 100.00

    def test_check_timeout_within_limit(self, policies):
        """Test check timeout passes within limit."""
        enforcer = PolicyEnforcer(policies)
        enforcer.start_context("ctx-1")

        # Should not raise (just started)
        enforcer.check_timeout("ctx-1", "strict")

    def test_check_timeout_exceeds_limit(self, policies):
        """Test check timeout raises when exceeded."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")

        # Simulate timeout by adjusting start time
        context.start_time = time.time() - 61  # 61 seconds ago

        with pytest.raises(PolicyViolation) as exc_info:
            enforcer.check_timeout("ctx-1", "strict")

        assert exc_info.value.constraint == "timeout_seconds"

    def test_check_timeout_no_policy(self, policies):
        """Test check timeout passes with no policy."""
        enforcer = PolicyEnforcer(policies)
        context = enforcer.start_context("ctx-1")
        context.start_time = time.time() - 1000

        # Should not raise
        enforcer.check_timeout("ctx-1", None)

    def test_check_timeout_no_context(self, policies):
        """Test check timeout passes with no context."""
        enforcer = PolicyEnforcer(policies)
        # Should not raise
        enforcer.check_timeout("missing", "strict")

    def test_multiple_contexts(self, policies):
        """Test managing multiple contexts."""
        enforcer = PolicyEnforcer(policies)

        ctx1 = enforcer.start_context("ctx-1")
        ctx2 = enforcer.start_context("ctx-2")

        ctx1.capability_calls = 5
        ctx2.capability_calls = 8

        context1 = enforcer.get_context("ctx-1")
        context2 = enforcer.get_context("ctx-2")
        assert context1 is not None
        assert context2 is not None
        assert context1.capability_calls == 5
        assert context2.capability_calls == 8

        enforcer.end_context("ctx-1")
        assert enforcer.get_context("ctx-1") is None
        context2_after = enforcer.get_context("ctx-2")
        assert context2_after is not None
        assert context2_after.capability_calls == 8
