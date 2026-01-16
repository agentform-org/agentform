"""Tests for workflow state management."""

import pytest

from agentform_runtime.state import WorkflowState


class TestWorkflowState:
    """Tests for WorkflowState class."""

    def test_init_empty(self):
        """Test initialization with no input data."""
        state = WorkflowState()
        assert state.input == {}
        assert state.state == {}

    def test_init_with_input(self):
        """Test initialization with input data."""
        input_data = {"question": "Hello", "count": 42}
        state = WorkflowState(input_data)
        assert state.input == input_data
        assert state.state == {}

    def test_set_and_get(self):
        """Test setting and getting state values."""
        state = WorkflowState()
        state.set("result", "success")
        state.set("count", 42)

        assert state.get("result") == "success"
        assert state.get("count") == 42

    def test_get_with_default(self):
        """Test getting non-existent key returns default."""
        state = WorkflowState()
        assert state.get("missing") is None
        assert state.get("missing", "default") == "default"

    def test_set_overwrites(self):
        """Test setting same key overwrites value."""
        state = WorkflowState()
        state.set("key", "value1")
        state.set("key", "value2")
        assert state.get("key") == "value2"

    def test_input_property(self):
        """Test input property returns input data."""
        input_data = {"x": 1, "y": 2}
        state = WorkflowState(input_data)
        assert state.input is input_data

    def test_state_property(self):
        """Test state property returns state dict."""
        state = WorkflowState()
        state.set("a", 1)
        state.set("b", 2)
        assert state.state == {"a": 1, "b": 2}


class TestWorkflowStateResolve:
    """Tests for expression resolution."""

    def test_resolve_non_expression(self):
        """Test resolving non-expression returns as-is."""
        state = WorkflowState()
        assert state.resolve("hello") == "hello"
        assert state.resolve("123") == "123"

    def test_resolve_input_simple(self):
        """Test resolving $input.field."""
        state = WorkflowState({"name": "Alice", "age": 30})
        assert state.resolve("$input.name") == "Alice"
        assert state.resolve("$input.age") == 30

    def test_resolve_state_simple(self):
        """Test resolving $state.key."""
        state = WorkflowState()
        state.set("result", "success")
        state.set("data", {"nested": "value"})

        assert state.resolve("$state.result") == "success"

    def test_resolve_nested_path(self):
        """Test resolving nested paths."""
        state = WorkflowState({"user": {"profile": {"name": "Bob"}}})
        state.set("response", {"data": {"items": [1, 2, 3]}})

        assert state.resolve("$input.user.profile.name") == "Bob"
        assert state.resolve("$state.response.data") == {"items": [1, 2, 3]}

    def test_resolve_invalid_root(self):
        """Test resolving invalid root raises KeyError."""
        state = WorkflowState()
        with pytest.raises(KeyError) as exc_info:
            state.resolve("$unknown.field")
        assert "unknown" in str(exc_info.value)

    def test_resolve_missing_path(self):
        """Test resolving missing path raises KeyError."""
        state = WorkflowState({"exists": "value"})
        with pytest.raises(KeyError):
            state.resolve("$input.missing")

    def test_resolve_invalid_expression(self):
        """Test resolving invalid expression raises KeyError."""
        state = WorkflowState()
        with pytest.raises(KeyError):
            state.resolve("$")

    def test_resolve_deep_nested_missing(self):
        """Test resolving deep nested missing path raises KeyError."""
        state = WorkflowState({"a": {"b": {"c": "value"}}})
        with pytest.raises(KeyError):
            state.resolve("$input.a.b.missing.deep")


class TestWorkflowStateResolveDict:
    """Tests for dictionary resolution."""

    def test_resolve_dict_simple(self):
        """Test resolving simple dictionary."""
        state = WorkflowState({"name": "Alice"})
        state.set("count", 10)

        data = {
            "greeting": "$input.name",
            "total": "$state.count",
        }

        result = state.resolve_dict(data)
        assert result == {"greeting": "Alice", "total": 10}

    def test_resolve_dict_mixed(self):
        """Test resolving dictionary with mixed values."""
        state = WorkflowState({"x": 1})

        data = {
            "expr": "$input.x",
            "literal": "hello",
            "number": 42,
            "boolean": True,
        }

        result = state.resolve_dict(data)
        assert result == {
            "expr": 1,
            "literal": "hello",
            "number": 42,
            "boolean": True,
        }

    def test_resolve_dict_nested(self):
        """Test resolving nested dictionary."""
        state = WorkflowState({"a": 1, "b": 2})

        data = {
            "outer": {
                "inner": "$input.a",
                "literal": "text",
            },
            "top": "$input.b",
        }

        result = state.resolve_dict(data)
        assert result == {
            "outer": {
                "inner": 1,
                "literal": "text",
            },
            "top": 2,
        }

    def test_resolve_dict_with_list(self):
        """Test resolving dictionary with list values."""
        state = WorkflowState({"items": ["a", "b", "c"]})
        state.set("extra", "d")

        data = {
            "list": ["$input.items", "$state.extra", "literal"],
        }

        result = state.resolve_dict(data)
        # List items that are strings get resolved
        assert result["list"][0] == ["a", "b", "c"]
        assert result["list"][1] == "d"
        assert result["list"][2] == "literal"


class TestWorkflowStateSerialization:
    """Tests for state serialization."""

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = WorkflowState({"input_key": "input_value"})
        state.set("state_key", "state_value")

        data = state.to_dict()

        assert data == {
            "input": {"input_key": "input_value"},
            "state": {"state_key": "state_value"},
        }

    def test_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            "input": {"a": 1, "b": 2},
            "state": {"x": "foo", "y": "bar"},
        }

        state = WorkflowState.from_dict(data)

        assert state.input == {"a": 1, "b": 2}
        assert state.state == {"x": "foo", "y": "bar"}

    def test_from_dict_partial(self):
        """Test creating state from partial dictionary."""
        state = WorkflowState.from_dict({})
        assert state.input == {}
        assert state.state == {}

        state = WorkflowState.from_dict({"input": {"a": 1}})
        assert state.input == {"a": 1}
        assert state.state == {}

    def test_roundtrip(self):
        """Test roundtrip serialization."""
        original = WorkflowState({"key": "value"})
        original.set("result", {"nested": [1, 2, 3]})

        data = original.to_dict()
        restored = WorkflowState.from_dict(data)

        assert restored.input == original.input
        assert restored.state == original.state


class TestEvaluateCondition:
    """Tests for condition evaluation."""

    def test_simple_equality(self):
        """Test simple equality comparison."""
        state = WorkflowState({"env": "prod"})
        assert state.evaluate_condition('$input.env == "prod"') is True
        assert state.evaluate_condition('$input.env == "dev"') is False

    def test_inequality(self):
        """Test inequality comparison."""
        state = WorkflowState({"status": "success"})
        assert state.evaluate_condition('$input.status != "error"') is True
        assert state.evaluate_condition('$input.status != "success"') is False

    def test_numeric_comparisons(self):
        """Test numeric comparison operators."""
        state = WorkflowState({"count": 5})
        assert state.evaluate_condition("$input.count > 3") is True
        assert state.evaluate_condition("$input.count > 5") is False
        assert state.evaluate_condition("$input.count >= 5") is True
        assert state.evaluate_condition("$input.count < 10") is True
        assert state.evaluate_condition("$input.count <= 5") is True

    def test_logical_and(self):
        """Test logical AND operator."""
        state = WorkflowState({"a": True, "b": True, "c": False})
        assert state.evaluate_condition("$input.a && $input.b") is True
        assert state.evaluate_condition("$input.a && $input.c") is False

    def test_logical_or(self):
        """Test logical OR operator."""
        state = WorkflowState({"a": True, "b": False, "c": False})
        assert state.evaluate_condition("$input.a || $input.b") is True
        assert state.evaluate_condition("$input.b || $input.c") is False

    def test_logical_not(self):
        """Test logical NOT operator."""
        state = WorkflowState({"flag": True})
        assert state.evaluate_condition("!$input.flag") is False
        state2 = WorkflowState({"flag": False})
        assert state2.evaluate_condition("!$input.flag") is True

    def test_combined_logical(self):
        """Test combined logical operators."""
        state = WorkflowState({"a": True, "b": False, "c": True})
        # a && c should be True
        assert state.evaluate_condition("$input.a && $input.c") is True
        # a && b should be False
        assert state.evaluate_condition("$input.a && $input.b") is False
        # (a && b) || c should be True
        assert state.evaluate_condition("$input.a && $input.b || $input.c") is True

    def test_state_ref_comparison(self):
        """Test state reference in comparisons."""
        state = WorkflowState()
        state.set("result", {"status": "success", "code": 200})
        assert state.evaluate_condition('$state.result.status == "success"') is True
        assert state.evaluate_condition("$state.result.code == 200") is True

    def test_boolean_value_direct(self):
        """Test evaluating boolean values directly."""
        state = WorkflowState({"enabled": True, "disabled": False})
        assert state.evaluate_condition("$input.enabled") is True
        assert state.evaluate_condition("$input.disabled") is False

    def test_string_truthiness(self):
        """Test string truthiness evaluation."""
        state = WorkflowState({"filled": "hello", "empty": ""})
        assert state.evaluate_condition("$input.filled") is True
        assert state.evaluate_condition("$input.empty") is False

    def test_complex_condition_with_state_and_input(self):
        """Test complex condition with both state and input."""
        state = WorkflowState({"threshold": 10})
        state.set("metrics", {"count": 15})
        assert state.evaluate_condition("$state.metrics.count > 10") is True
        assert state.evaluate_condition("$state.metrics.count > 20") is False
