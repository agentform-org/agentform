"""Workflow state management."""

import re
from typing import Any


class WorkflowState:
    """Manages state during workflow execution.

    State is accessed via expressions like:
    - $input.field - Input data passed to workflow
    - $state.step_id - Output from a previous step
    - $state.step_id.field - Nested field access
    """

    def __init__(self, input_data: dict[str, Any] | None = None):
        """Initialize workflow state.

        Args:
            input_data: Initial input data for $input references
        """
        self._input = input_data or {}
        self._state: dict[str, Any] = {}

    @property
    def input(self) -> dict[str, Any]:
        """Get input data."""
        return self._input

    @property
    def state(self) -> dict[str, Any]:
        """Get all state."""
        return self._state

    def set(self, key: str, value: Any) -> None:
        """Set a state value.

        Args:
            key: State key (typically step ID or save_as name)
            value: Value to store
        """
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            State value or default
        """
        return self._state.get(key, default)

    def resolve(self, expr: str) -> Any:
        """Resolve an expression to its value.

        Args:
            expr: Expression like "$input.field" or "$state.step.field"

        Returns:
            Resolved value

        Raises:
            KeyError: If path not found
        """
        if not expr.startswith("$"):
            return expr

        # Parse expression
        parts = expr[1:].split(".")
        if not parts:
            raise KeyError(f"Invalid expression: {expr}")

        root = parts[0]
        path = parts[1:]

        if root == "input":
            value = self._input
        elif root == "state":
            value = self._state
        else:
            raise KeyError(f"Unknown root '{root}' in expression: {expr}")

        # Navigate path
        for part in path:
            if isinstance(value, dict):
                if part not in value:
                    raise KeyError(f"Path '{'.'.join(parts[:parts.index(part)+1])}' not found")
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise KeyError(f"Cannot access '{part}' on {type(value)}")

        return value

    def resolve_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve all expressions in a dictionary.

        Args:
            data: Dictionary with possible expression values

        Returns:
            Dictionary with resolved values
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.resolve(value)
            elif isinstance(value, dict):
                result[key] = self.resolve_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.resolve(v) if isinstance(v, str) else v for v in value
                ]
            else:
                result[key] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "input": self._input,
            "state": self._state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowState":
        """Create state from dictionary."""
        instance = cls(data.get("input", {}))
        instance._state = data.get("state", {})
        return instance

