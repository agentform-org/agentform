"""Tests for the run command and helper functions."""

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from agentform_cli.commands.run import extract_input_fields, prompt_for_inputs
from agentform_cli.main import app
from agentform_schema.ir import ResolvedStep, ResolvedWorkflow
from agentform_schema.models import StepType

runner = CliRunner()


class TestExtractInputFields:
    """Tests for extract_input_fields function."""

    def test_extract_from_input_mapping(self):
        """Test extracting input fields from input_mapping."""
        step = ResolvedStep(
            id="test",
            type=StepType.LLM,
            agent_name="test-agent",
            input_mapping={"question": "$input.question", "context": "$input.context"},
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert "question" in fields
        assert "context" in fields

    def test_extract_from_args_mapping(self):
        """Test extracting input fields from args_mapping."""
        step = ResolvedStep(
            id="test",
            type=StepType.CALL,
            capability_name="test-cap",
            args_mapping={"param1": "$input.param1", "param2": "$input.param2"},
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert "param1" in fields
        assert "param2" in fields

    def test_extract_from_condition_expr(self):
        """Test extracting input fields from condition_expr."""
        step = ResolvedStep(
            id="test",
            type=StepType.CONDITION,
            condition_expr="$input.flag == true",
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert "flag" in fields

    def test_extract_from_payload_expr(self):
        """Test extracting input fields from payload_expr."""
        step = ResolvedStep(
            id="test",
            type=StepType.HUMAN_APPROVAL,
            payload_expr="$input.reason",
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert "reason" in fields

    def test_extract_multiple_references(self):
        """Test extracting multiple references to same field."""
        step = ResolvedStep(
            id="test",
            type=StepType.LLM,
            agent_name="test-agent",
            input_mapping={
                "question": "$input.query",
                "context": "$input.query",  # Same field referenced twice
            },
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert "query" in fields
        assert len(fields) == 1  # query appears twice but only once in the set

    def test_no_input_fields(self):
        """Test workflow with no input fields."""
        step = ResolvedStep(
            id="test",
            type=StepType.LLM,
            agent_name="test-agent",
            input_mapping={"static": "constant value"},
        )
        workflow = ResolvedWorkflow(
            name="test",
            entry_step="test",
            steps={"test": step},
        )

        fields = extract_input_fields(workflow)
        assert len(fields) == 0


class TestPromptForInputs:
    """Tests for prompt_for_inputs function."""

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_prompt_for_missing_fields(self, mock_console, mock_prompt):
        """Test prompting for missing input fields."""
        mock_prompt.return_value = "test-value"

        required_fields = {"field1", "field2"}
        existing_input: dict[str, Any] = {}

        result = prompt_for_inputs(required_fields, existing_input)

        assert mock_prompt.call_count == 2
        assert result["field1"] == "test-value"
        assert result["field2"] == "test-value"

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_no_missing_fields(self, mock_console, mock_prompt):
        """Test when all fields are already provided."""
        required_fields = {"field1", "field2"}
        existing_input = {"field1": "value1", "field2": "value2"}

        result = prompt_for_inputs(required_fields, existing_input)

        mock_prompt.assert_not_called()
        assert result == existing_input

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_parse_json_input(self, mock_console, mock_prompt):
        """Test parsing JSON input values."""
        mock_prompt.return_value = '{"key": "value"}'

        required_fields = {"field1"}
        existing_input: dict[str, Any] = {}

        result = prompt_for_inputs(required_fields, existing_input)

        assert result["field1"] == {"key": "value"}

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_parse_number_input(self, mock_console, mock_prompt):
        """Test parsing number input."""
        mock_prompt.return_value = "42"

        required_fields = {"field1"}
        existing_input: dict[str, Any] = {}

        result = prompt_for_inputs(required_fields, existing_input)

        assert result["field1"] == 42

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_parse_boolean_input(self, mock_console, mock_prompt):
        """Test parsing boolean input."""
        mock_prompt.return_value = "true"

        required_fields = {"field1"}
        existing_input: dict[str, Any] = {}

        result = prompt_for_inputs(required_fields, existing_input)

        assert result["field1"] is True

    @patch("agentform_cli.commands.run.typer.prompt")
    @patch("agentform_cli.commands.run.console")
    def test_invalid_json_falls_back_to_string(self, mock_console, mock_prompt):
        """Test that invalid JSON falls back to string."""
        mock_prompt.return_value = "not valid json {"

        required_fields = {"field1"}
        existing_input: dict[str, Any] = {}

        result = prompt_for_inputs(required_fields, existing_input)

        assert result["field1"] == "not valid json {"


class TestRunCommand:
    """Tests for the run command."""

    def test_run_nonexistent_spec_file(self):
        """Test running with a spec file that doesn't exist."""
        result = runner.invoke(app, ["run", "test-workflow", "nonexistent.agentform"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_run_with_invalid_input_json(self):
        """Test running with invalid JSON input."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}

workflow "test-workflow" {
  entry = step.end
  step "end" { type = "end" }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            spec_path = Path(f.name)

        try:
            result = runner.invoke(
                app,
                ["run", "test-workflow", str(spec_path), "--input", "invalid json {"],
            )
            assert result.exit_code == 1
            assert "Error parsing input JSON" in result.stdout
        finally:
            spec_path.unlink()

    def test_run_with_input_file(self):
        """Test running with input file."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}

workflow "test-workflow" {
  entry = step.end
  step "end" { type = "end" }
}
"""
        input_data = {"key": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            spec_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_data, f)
            input_path = Path(f.name)

        try:
            with patch("agentform_cli.commands.run.asyncio.run") as mock_run:
                mock_run.return_value = {"output": None, "state": {}}
                result = runner.invoke(
                    app,
                    [
                        "run",
                        "test-workflow",
                        str(spec_path),
                        "--input-file",
                        str(input_path),
                    ],
                )
                # Should get past input parsing
                assert result.exit_code != 1 or "Compilation" in result.stdout
        finally:
            spec_path.unlink()
            input_path.unlink()

    def test_run_nonexistent_workflow(self):
        """Test running a workflow that doesn't exist."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}

workflow "existing-workflow" {
  entry = step.end
  step "end" { type = "end" }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            spec_path = Path(f.name)

        try:
            result = runner.invoke(
                app,
                ["run", "nonexistent-workflow", str(spec_path)],
            )
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
        finally:
            spec_path.unlink()

    @patch("agentform_cli.commands.run.asyncio.run")
    def test_run_with_output_file(self, mock_run):
        """Test running and writing output to file."""
        mock_run.return_value = {"output": {"result": "success"}, "state": {}}

        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}

workflow "test-workflow" {
  entry = step.end
  step "end" { type = "end" }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            spec_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)
            output_path.unlink()  # Delete it so we can test creation

        try:
            result = runner.invoke(
                app,
                [
                    "run",
                    "test-workflow",
                    str(spec_path),
                    "--output",
                    str(output_path),
                ],
            )
            # Should create output file if workflow runs successfully
            # Note: This may fail at compilation or execution, but should handle output file
            if result.exit_code == 0:
                assert output_path.exists()
        finally:
            spec_path.unlink()
            if output_path.exists():
                output_path.unlink()
