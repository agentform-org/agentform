"""Type definitions for agentform-api."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowResult:
    """Result of a workflow execution.

    Attributes:
        output: The final output from the workflow
        state: Full workflow state dictionary containing input and all step outputs
        trace: Execution trace for debugging and observability
    """

    output: dict[str, Any] | Any = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
