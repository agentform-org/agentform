"""Workflow state management."""

from typing import Any, cast

# Import AST types for expression evaluation (optional dependency)
try:
    from agentform_compiler.agentform_ast import (
        AndExpr,
        ComparisonExpr,
        ConditionalExpr,
        NotExpr,
        OrExpr,
        Reference,
        StateRef,
    )

    _HAS_AST_TYPES = True
except ImportError:
    _HAS_AST_TYPES = False


class ExpressionError(Exception):
    """Error during expression evaluation."""

    pass


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

    def resolve(self, expr: Any) -> Any:
        """Resolve an expression to its value.

        Args:
            expr: Expression - can be:
                - A string like "$input.field" or "$state.step.field"
                - A primitive value (str, int, float, bool)
                - An AST expression node (ConditionalExpr, ComparisonExpr, etc.)

        Returns:
            Resolved value

        Raises:
            KeyError: If path not found
            ExpressionError: If expression evaluation fails
        """
        # Handle AST expression types if available
        if _HAS_AST_TYPES:
            if isinstance(expr, ConditionalExpr):
                return self._eval_conditional(expr)
            elif isinstance(expr, ComparisonExpr):
                return self._eval_comparison(expr)
            elif isinstance(expr, AndExpr):
                return self._eval_and(expr)
            elif isinstance(expr, OrExpr):
                return self._eval_or(expr)
            elif isinstance(expr, NotExpr):
                return self._eval_not(expr)
            elif isinstance(expr, StateRef):
                return self._resolve_state_ref(expr.path)
            elif isinstance(expr, Reference):
                # Static references - return as-is for now
                return expr.path

        # Handle string expressions
        if isinstance(expr, str):
            if expr.startswith("$"):
                return self._resolve_state_ref(expr)
            return expr

        # Handle primitives (int, float, bool, etc.)
        return expr

    def _resolve_state_ref(self, expr: str) -> Any:
        """Resolve a state reference string like $input.field or $state.step.field."""
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
                    raise KeyError(f"Path '{'.'.join(parts[: parts.index(part) + 1])}' not found")
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise KeyError(f"Cannot access '{part}' on {type(value)}")

        return value

    def _eval_conditional(self, expr: Any) -> Any:
        """Evaluate a conditional expression: condition ? true_val : false_val."""
        condition = self.resolve(expr.condition)
        if self._to_bool(condition):
            return self.resolve(expr.true_value)
        else:
            return self.resolve(expr.false_value)

    def _eval_comparison(self, expr: Any) -> bool:
        """Evaluate a comparison expression: left op right."""
        left = self.resolve(expr.left)
        right = self.resolve(expr.right)
        op = expr.operator

        if op == "==":
            return cast("bool", left == right)
        elif op == "!=":
            return cast("bool", left != right)
        elif op == "<":
            return cast("bool", left < right)
        elif op == ">":
            return cast("bool", left > right)
        elif op == "<=":
            return cast("bool", left <= right)
        elif op == ">=":
            return cast("bool", left >= right)
        else:
            raise ExpressionError(f"Unknown comparison operator: {op}")

    def _eval_and(self, expr: Any) -> bool:
        """Evaluate a logical AND expression."""
        return all(self._to_bool(self.resolve(operand)) for operand in expr.operands)

    def _eval_or(self, expr: Any) -> bool:
        """Evaluate a logical OR expression."""
        return any(self._to_bool(self.resolve(operand)) for operand in expr.operands)

    def _eval_not(self, expr: Any) -> bool:
        """Evaluate a logical NOT expression."""
        return not self._to_bool(self.resolve(expr.operand))

    def _to_bool(self, value: Any) -> bool:
        """Convert a value to boolean for condition evaluation."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Empty string is falsy, "false" is falsy (case-insensitive)
            return value.lower() not in ("false", "no", "0", "")
        if value is None:
            return False
        if isinstance(value, int | float):
            return value != 0
        # Default: truthy if not None
        return bool(value)

    def evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition expression string and return a boolean.

        Supports:
        - Simple comparisons: $state.x == "value", $state.x != "value"
        - Boolean checks: $state.x, !$state.x
        - Logical operators: $state.x && $state.y, $state.x || $state.y
        - Conditional expressions: $state.x ? "yes" : "no" (evaluates to bool)

        Args:
            condition: Condition expression string

        Returns:
            Boolean result of the condition evaluation
        """
        condition = condition.strip()

        # Handle conditional expressions (ternary)
        if "?" in condition and ":" in condition:
            # Parse: condition ? true_val : false_val
            # Find the ? that marks the ternary (not inside quotes)
            q_pos = self._find_ternary_operator(condition)
            if q_pos > 0:
                cond_part = condition[:q_pos].strip()
                rest = condition[q_pos + 1 :]
                c_pos = self._find_colon(rest)
                if c_pos > 0:
                    cond_result = self.evaluate_condition(cond_part)
                    return cond_result

        # Handle logical OR (lowest precedence)
        if "||" in condition:
            parts = self._split_logical(condition, "||")
            if len(parts) > 1:
                return any(self.evaluate_condition(p.strip()) for p in parts)

        # Handle logical AND
        if "&&" in condition:
            parts = self._split_logical(condition, "&&")
            if len(parts) > 1:
                return all(self.evaluate_condition(p.strip()) for p in parts)

        # Handle NOT
        if condition.startswith("!"):
            return not self.evaluate_condition(condition[1:].strip())

        # Handle comparisons
        for op in ["==", "!=", "<=", ">=", "<", ">"]:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = self.resolve(parts[0].strip())
                    right = parts[1].strip().strip('"').strip("'")
                    if op == "==":
                        return str(left) == right
                    elif op == "!=":
                        return str(left) != right
                    elif op == "<":
                        return float(left) < float(right)
                    elif op == ">":
                        return float(left) > float(right)
                    elif op == "<=":
                        return float(left) <= float(right)
                    elif op == ">=":
                        return float(left) >= float(right)

        # Simple boolean evaluation
        return self._to_bool(self.resolve(condition))

    def _find_ternary_operator(self, expr: str) -> int:
        """Find position of ? in ternary expression (not inside quotes)."""
        in_quotes = False
        quote_char = None
        for i, c in enumerate(expr):
            if c in ('"', "'") and (i == 0 or expr[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = c
                elif c == quote_char:
                    in_quotes = False
            elif c == "?" and not in_quotes:
                return i
        return -1

    def _find_colon(self, expr: str) -> int:
        """Find position of : in ternary expression (not inside quotes)."""
        in_quotes = False
        quote_char = None
        for i, c in enumerate(expr):
            if c in ('"', "'") and (i == 0 or expr[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = c
                elif c == quote_char:
                    in_quotes = False
            elif c == ":" and not in_quotes:
                return i
        return -1

    def _split_logical(self, expr: str, op: str) -> list[str]:
        """Split expression by logical operator (not inside quotes)."""
        parts: list[str] = []
        current = ""
        in_quotes = False
        quote_char = None
        i = 0
        while i < len(expr):
            c = expr[i]
            if c in ('"', "'") and (i == 0 or expr[i - 1] != "\\"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = c
                elif c == quote_char:
                    in_quotes = False
                current += c
            elif not in_quotes and expr[i:].startswith(op):
                parts.append(current)
                current = ""
                i += len(op) - 1
            else:
                current += c
            i += 1
        if current:
            parts.append(current)
        return parts

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
                result[key] = [self.resolve(v) if isinstance(v, str) else v for v in value]
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
