"""Tests for execution tracing."""

import json
import time

from agentform_runtime.tracing import EventType, TraceEvent, Tracer


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types(self):
        """Test all event types exist."""
        assert EventType.WORKFLOW_START.value == "workflow_start"
        assert EventType.WORKFLOW_END.value == "workflow_end"
        assert EventType.WORKFLOW_ERROR.value == "workflow_error"
        assert EventType.STEP_START.value == "step_start"
        assert EventType.STEP_END.value == "step_end"
        assert EventType.STEP_ERROR.value == "step_error"
        assert EventType.LLM_CALL.value == "llm_call"
        assert EventType.CAPABILITY_CALL.value == "capability_call"
        assert EventType.APPROVAL_REQUEST.value == "approval_request"
        assert EventType.APPROVAL_RESPONSE.value == "approval_response"
        assert EventType.POLICY_CHECK.value == "policy_check"
        assert EventType.STATE_UPDATE.value == "state_update"


class TestTraceEvent:
    """Tests for TraceEvent dataclass."""

    def test_minimal_event(self):
        """Test creating minimal event."""
        event = TraceEvent(
            type=EventType.STEP_START,
            timestamp=time.time(),
            workflow_name="test",
        )
        assert event.type == EventType.STEP_START
        assert event.workflow_name == "test"
        assert event.step_id is None
        assert event.data == {}
        assert event.trace_id is not None  # Auto-generated

    def test_full_event(self):
        """Test creating event with all fields."""
        event = TraceEvent(
            type=EventType.LLM_CALL,
            timestamp=12345.0,
            workflow_name="main",
            step_id="process",
            data={"model": "gpt-4", "tokens": 100},
            trace_id="trace-123",
            parent_id="parent-456",
        )
        assert event.type == EventType.LLM_CALL
        assert event.timestamp == 12345.0
        assert event.step_id == "process"
        assert event.data["model"] == "gpt-4"
        assert event.trace_id == "trace-123"
        assert event.parent_id == "parent-456"


class TestTracer:
    """Tests for Tracer class."""

    def test_init(self):
        """Test tracer initialization."""
        tracer = Tracer("test-workflow")
        assert tracer.workflow_name == "test-workflow"
        assert tracer.trace_id is not None
        assert len(tracer.events) == 0
        assert tracer._start_time > 0

    def test_emit(self):
        """Test emitting basic event."""
        tracer = Tracer("test")
        event = tracer.emit(EventType.STEP_START, step_id="step1")

        assert event.type == EventType.STEP_START
        assert event.step_id == "step1"
        assert event.workflow_name == "test"
        assert event.trace_id == tracer.trace_id
        assert len(tracer.events) == 1

    def test_emit_with_data(self):
        """Test emitting event with data."""
        tracer = Tracer("test")
        event = tracer.emit(
            EventType.LLM_CALL,
            step_id="llm",
            data={"model": "gpt-4", "tokens": 50},
        )

        assert event.data["model"] == "gpt-4"
        assert event.data["tokens"] == 50

    def test_workflow_start(self):
        """Test workflow start event."""
        tracer = Tracer("test")
        input_data = {"question": "Hello?"}
        event = tracer.workflow_start(input_data)

        assert event.type == EventType.WORKFLOW_START
        assert event.data["input"] == input_data

    def test_workflow_end(self):
        """Test workflow end event."""
        tracer = Tracer("test")
        output = {"answer": "World!"}
        event = tracer.workflow_end(output)

        assert event.type == EventType.WORKFLOW_END
        assert event.data["output"] == output
        assert "duration_seconds" in event.data

    def test_workflow_error(self):
        """Test workflow error event."""
        tracer = Tracer("test")
        error = ValueError("Something went wrong")
        event = tracer.workflow_error(error)

        assert event.type == EventType.WORKFLOW_ERROR
        assert "Something went wrong" in event.data["error"]
        assert event.data["error_type"] == "ValueError"

    def test_step_start(self):
        """Test step start event."""
        tracer = Tracer("test")
        event = tracer.step_start("process", "llm")

        assert event.type == EventType.STEP_START
        assert event.step_id == "process"
        assert event.data["step_type"] == "llm"

    def test_step_end(self):
        """Test step end event."""
        tracer = Tracer("test")
        event = tracer.step_end("process", {"result": "success"})

        assert event.type == EventType.STEP_END
        assert event.step_id == "process"
        assert event.data["output"] == {"result": "success"}

    def test_step_error(self):
        """Test step error event."""
        tracer = Tracer("test")
        error = RuntimeError("Step failed")
        event = tracer.step_error("process", error)

        assert event.type == EventType.STEP_ERROR
        assert event.step_id == "process"
        assert "Step failed" in event.data["error"]
        assert event.data["error_type"] == "RuntimeError"

    def test_llm_call(self):
        """Test LLM call event."""
        tracer = Tracer("test")
        event = tracer.llm_call(
            step_id="llm",
            model="gpt-4",
            prompt="Hello?",
            response="Hi there!",
            tokens=25,
        )

        assert event.type == EventType.LLM_CALL
        assert event.step_id == "llm"
        assert event.data["model"] == "gpt-4"
        assert event.data["prompt_preview"] == "Hello?"
        assert event.data["response_preview"] == "Hi there!"
        assert event.data["tokens"] == 25

    def test_llm_call_truncates_long_content(self):
        """Test LLM call truncates long prompts/responses."""
        tracer = Tracer("test")
        long_prompt = "x" * 1000
        long_response = "y" * 1000

        event = tracer.llm_call(
            step_id="llm",
            model="gpt-4",
            prompt=long_prompt,
            response=long_response,
        )

        assert len(event.data["prompt_preview"]) == 500
        assert len(event.data["response_preview"]) == 500

    def test_capability_call(self):
        """Test capability call event."""
        tracer = Tracer("test")
        event = tracer.capability_call(
            step_id="call",
            capability="read_file",
            args={"path": "/tmp/test"},
            result="File content",
        )

        assert event.type == EventType.CAPABILITY_CALL
        assert event.step_id == "call"
        assert event.data["capability"] == "read_file"
        assert event.data["args"] == {"path": "/tmp/test"}
        assert "File content" in event.data["result_preview"]

    def test_capability_call_truncates_result(self):
        """Test capability call truncates long results."""
        tracer = Tracer("test")
        long_result = "z" * 1000

        event = tracer.capability_call(
            step_id="call",
            capability="read_file",
            args={},
            result=long_result,
        )

        assert len(event.data["result_preview"]) == 500

    def test_approval_request(self):
        """Test approval request event."""
        tracer = Tracer("test")
        event = tracer.approval_request("approve", {"changes": ["a", "b"]})

        assert event.type == EventType.APPROVAL_REQUEST
        assert event.step_id == "approve"
        assert event.data["payload"] == {"changes": ["a", "b"]}

    def test_approval_response(self):
        """Test approval response event."""
        tracer = Tracer("test")

        approved = tracer.approval_response("approve", True)
        assert approved.type == EventType.APPROVAL_RESPONSE
        assert approved.data["approved"] is True

        rejected = tracer.approval_response("approve", False)
        assert rejected.data["approved"] is False

    def test_events_property(self):
        """Test events property returns all events."""
        tracer = Tracer("test")
        tracer.emit(EventType.STEP_START, step_id="s1")
        tracer.emit(EventType.STEP_END, step_id="s1")

        assert len(tracer.events) == 2

    def test_to_json(self):
        """Test JSON export."""
        tracer = Tracer("test-workflow")
        tracer.workflow_start({"input": "data"})
        tracer.step_start("step1", "llm")
        tracer.step_end("step1", {"output": "result"})
        tracer.workflow_end({"final": "output"})

        json_str = tracer.to_json()
        data = json.loads(json_str)

        assert data["trace_id"] == tracer.trace_id
        assert data["workflow_name"] == "test-workflow"
        assert len(data["events"]) == 4

        # Check event structure
        event = data["events"][0]
        assert event["type"] == "workflow_start"
        assert "timestamp" in event
        assert "data" in event

    def test_to_json_is_valid_json(self):
        """Test that to_json produces valid JSON."""
        tracer = Tracer("test")
        tracer.emit(EventType.STEP_START)

        json_str = tracer.to_json()
        # Should not raise
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_multiple_workflow_runs(self):
        """Test multiple tracers have different trace IDs."""
        tracer1 = Tracer("workflow1")
        tracer2 = Tracer("workflow2")

        assert tracer1.trace_id != tracer2.trace_id
