"""Tests for Agentform client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from agentform_api import Agentform, CompilationError, WorkflowError, WorkflowResult


class TestAgentformFromPath:
    """Tests for Agentform.from_path()."""

    def test_from_path_directory_not_found(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(CompilationError, match="Failed to compile spec"):
            Agentform.from_path("/nonexistent/path/")

    def test_from_path_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(CompilationError, match="Failed to compile spec"):
            Agentform.from_path("/nonexistent/file.agentform")

    @patch("agentform_api.client.compile_file")
    def test_from_path_with_variables(self, mock_compile):
        """Test that variables are passed to compiler."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {}
        mock_compile.return_value = mock_spec

        variables = {"api_key": "test-key", "model": "gpt-4o"}
        Agentform.from_path("test.agentform", variables=variables)

        mock_compile.assert_called_once()
        call_kwargs = mock_compile.call_args
        assert call_kwargs.kwargs["variables"] == variables

    @patch("agentform_api.client.compile_file")
    def test_from_path_with_approval_handler(self, mock_compile):
        """Test that approval handler is stored."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {}
        mock_compile.return_value = mock_spec

        mock_handler = MagicMock()
        agentform = Agentform.from_path("test.agentform", approval_handler=mock_handler)

        assert agentform._approval_handler == mock_handler

    @patch("agentform_api.client.compile_file")
    def test_from_path_with_verbose(self, mock_compile):
        """Test that verbose flag is stored."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {}
        mock_compile.return_value = mock_spec

        agentform = Agentform.from_path("test.agentform", verbose=True)

        assert agentform._verbose is True


class TestAgentformProperties:
    """Tests for Agentform properties."""

    @patch("agentform_api.client.compile_file")
    def test_workflows_property(self, mock_compile):
        """Test workflows property returns workflow names."""
        mock_spec = MagicMock()
        mock_spec.workflows = {"ask": MagicMock(), "process": MagicMock()}
        mock_spec.agents = {}
        mock_compile.return_value = mock_spec

        agentform = Agentform.from_path("test.agentform")

        assert set(agentform.workflows) == {"ask", "process"}

    @patch("agentform_api.client.compile_file")
    def test_agents_property(self, mock_compile):
        """Test agents property returns agent names."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {"assistant": MagicMock(), "reviewer": MagicMock()}
        mock_compile.return_value = mock_spec

        agentform = Agentform.from_path("test.agentform")

        assert set(agentform.agents) == {"assistant", "reviewer"}

    @patch("agentform_api.client.compile_file")
    def test_repr(self, mock_compile):
        """Test string representation."""
        mock_spec = MagicMock()
        mock_spec.workflows = {"w1": MagicMock(), "w2": MagicMock()}
        mock_spec.agents = {"a1": MagicMock()}
        mock_compile.return_value = mock_spec

        agentform = Agentform.from_path("test.agentform")

        assert repr(agentform) == "<Agentform workflows=2 agents=1>"


class TestAgentformRunWorkflow:
    """Tests for Agentform.run_workflow()."""

    @pytest.fixture
    def mock_agentform(self):
        """Create Agentform instance with mocked spec."""
        mock_spec = MagicMock()
        mock_spec.workflows = {"test_workflow": MagicMock()}
        mock_spec.agents = {}
        mock_spec.providers = {}
        mock_spec.policies = {}
        mock_spec.servers = {}
        return Agentform(mock_spec)

    @patch("agentform_api.client.WorkflowEngine")
    async def test_run_workflow_success(self, mock_engine_class, mock_agentform):
        """Test successful workflow execution."""
        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(
            return_value={
                "output": {"answer": "Paris"},
                "state": {"input": {"question": "Capital?"}, "state": {}},
                "trace": {"events": []},
            }
        )
        mock_engine_class.return_value = mock_engine

        result = await mock_agentform.run_workflow(
            "test_workflow", input_data={"question": "Capital?"}
        )

        assert isinstance(result, WorkflowResult)
        assert result.output == {"answer": "Paris"}
        assert "input" in result.state
        mock_engine.run.assert_called_once_with("test_workflow", {"question": "Capital?"})

    @patch("agentform_api.client.WorkflowEngine")
    async def test_run_workflow_no_input(self, mock_engine_class, mock_agentform):
        """Test workflow execution without input data."""
        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(return_value={"output": None, "state": {}, "trace": {}})
        mock_engine_class.return_value = mock_engine

        result = await mock_agentform.run_workflow("test_workflow")

        mock_engine.run.assert_called_once_with("test_workflow", None)
        assert result.output is None

    @patch("agentform_api.client.WorkflowEngine")
    async def test_run_workflow_error(self, mock_engine_class, mock_agentform):
        """Test workflow execution failure."""
        from agentform_runtime.engine import WorkflowError as RuntimeError

        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(side_effect=RuntimeError("Step failed"))
        mock_engine_class.return_value = mock_engine

        with pytest.raises(WorkflowError, match="Workflow 'test_workflow' failed"):
            await mock_agentform.run_workflow("test_workflow")

    @patch("agentform_api.client.WorkflowEngine")
    async def test_run_workflow_reuses_engine(self, mock_engine_class, mock_agentform):
        """Test that engine is reused across workflow runs."""
        mock_engine = MagicMock()
        mock_engine.run = AsyncMock(return_value={"output": None, "state": {}, "trace": {}})
        mock_engine_class.return_value = mock_engine

        await mock_agentform.run_workflow("test_workflow")
        await mock_agentform.run_workflow("test_workflow")

        # Engine should only be created once
        assert mock_engine_class.call_count == 1


class TestAgentformContextManager:
    """Tests for async context manager."""

    @patch("agentform_api.client.compile_file")
    async def test_context_manager(self, mock_compile):
        """Test async context manager."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {}
        mock_compile.return_value = mock_spec

        async with Agentform.from_path("test.agentform") as agentform:
            assert isinstance(agentform, Agentform)

        # After exiting, engine should be None
        assert agentform._engine is None

    @patch("agentform_api.client.compile_file")
    async def test_close(self, mock_compile):
        """Test close method."""
        mock_spec = MagicMock()
        mock_spec.workflows = {}
        mock_spec.agents = {}
        mock_spec.providers = {}
        mock_spec.policies = {}
        mock_spec.servers = {}
        mock_compile.return_value = mock_spec

        agentform = Agentform.from_path("test.agentform")
        # Create engine
        agentform._get_engine()
        assert agentform._engine is not None

        await agentform.close()
        assert agentform._engine is None


class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_default_values(self):
        """Test default values for WorkflowResult."""
        result = WorkflowResult()
        assert result.output == {}
        assert result.state == {}
        assert result.trace == {}

    def test_with_values(self):
        """Test WorkflowResult with values."""
        result = WorkflowResult(
            output={"answer": "42"},
            state={"input": {}, "state": {"step1": "done"}},
            trace={"events": [1, 2, 3]},
        )
        assert result.output == {"answer": "42"}
        assert result.state["state"]["step1"] == "done"
        assert len(result.trace["events"]) == 3

    def test_output_can_be_any_type(self):
        """Test that output can be any type."""
        result = WorkflowResult(output="string output")
        assert result.output == "string output"

        result = WorkflowResult(output=None)
        assert result.output is None
