"""Execution tracing and audit logging."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    """Types of trace events."""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    WORKFLOW_ERROR = "workflow_error"
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_ERROR = "step_error"
    LLM_CALL = "llm_call"
    CAPABILITY_CALL = "capability_call"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    POLICY_CHECK = "policy_check"
    STATE_UPDATE = "state_update"


@dataclass
class TraceEvent:
    """A single trace event."""

    type: EventType
    timestamp: float
    workflow_name: str
    step_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None


class Tracer:
    """Collects and manages trace events."""

    def __init__(self, workflow_name: str):
        """Initialize tracer.

        Args:
            workflow_name: Name of the workflow being traced
        """
        self.workflow_name = workflow_name
        self.trace_id = str(uuid4())
        self._events: list[TraceEvent] = []
        self._start_time = time.time()

    def emit(
        self,
        event_type: EventType,
        step_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        """Emit a trace event.

        Args:
            event_type: Type of event
            step_id: Optional step ID
            data: Optional event data

        Returns:
            The created event
        """
        event = TraceEvent(
            type=event_type,
            timestamp=time.time(),
            workflow_name=self.workflow_name,
            step_id=step_id,
            data=data or {},
            trace_id=self.trace_id,
        )
        self._events.append(event)
        return event

    def workflow_start(self, input_data: dict[str, Any]) -> TraceEvent:
        """Record workflow start."""
        return self.emit(
            EventType.WORKFLOW_START,
            data={"input": input_data},
        )

    def workflow_end(self, output: Any) -> TraceEvent:
        """Record workflow end."""
        return self.emit(
            EventType.WORKFLOW_END,
            data={
                "output": output,
                "duration_seconds": time.time() - self._start_time,
            },
        )

    def workflow_error(self, error: Exception) -> TraceEvent:
        """Record workflow error."""
        return self.emit(
            EventType.WORKFLOW_ERROR,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

    def step_start(self, step_id: str, step_type: str) -> TraceEvent:
        """Record step start."""
        return self.emit(
            EventType.STEP_START,
            step_id=step_id,
            data={"step_type": step_type},
        )

    def step_end(self, step_id: str, output: Any) -> TraceEvent:
        """Record step end."""
        return self.emit(
            EventType.STEP_END,
            step_id=step_id,
            data={"output": output},
        )

    def step_error(self, step_id: str, error: Exception) -> TraceEvent:
        """Record step error."""
        return self.emit(
            EventType.STEP_ERROR,
            step_id=step_id,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

    def llm_call(
        self,
        step_id: str,
        model: str,
        prompt: str,
        response: str,
        tokens: int | None = None,
    ) -> TraceEvent:
        """Record LLM call."""
        return self.emit(
            EventType.LLM_CALL,
            step_id=step_id,
            data={
                "model": model,
                "prompt_preview": prompt[:500] if len(prompt) > 500 else prompt,
                "response_preview": response[:500] if len(response) > 500 else response,
                "tokens": tokens,
            },
        )

    def capability_call(
        self,
        step_id: str,
        capability: str,
        args: dict[str, Any],
        result: Any,
    ) -> TraceEvent:
        """Record capability call."""
        return self.emit(
            EventType.CAPABILITY_CALL,
            step_id=step_id,
            data={
                "capability": capability,
                "args": args,
                "result_preview": str(result)[:500],
            },
        )

    def approval_request(self, step_id: str, payload: Any) -> TraceEvent:
        """Record approval request."""
        return self.emit(
            EventType.APPROVAL_REQUEST,
            step_id=step_id,
            data={"payload": payload},
        )

    def approval_response(self, step_id: str, approved: bool) -> TraceEvent:
        """Record approval response."""
        return self.emit(
            EventType.APPROVAL_RESPONSE,
            step_id=step_id,
            data={"approved": approved},
        )

    @property
    def events(self) -> list[TraceEvent]:
        """Get all events."""
        return self._events

    def to_json(self) -> str:
        """Export trace as JSON."""
        return json.dumps(
            {
                "trace_id": self.trace_id,
                "workflow_name": self.workflow_name,
                "events": [
                    {
                        "type": e.type.value,
                        "timestamp": e.timestamp,
                        "step_id": e.step_id,
                        "data": e.data,
                    }
                    for e in self._events
                ],
            },
            indent=2,
        )
