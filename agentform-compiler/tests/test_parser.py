"""Tests for Agentform YAML parser."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler.parser import ParseError, parse_yaml, parse_yaml_file
from agentform_schema.models import SideEffect, StepType


class TestParseYaml:
    """Tests for parse_yaml function."""

    def test_minimal_valid_yaml(self):
        """Test parsing minimal valid YAML."""
        yaml_content = """
version: "0.1"
project:
  name: test-project
"""
        spec = parse_yaml(yaml_content)
        assert spec.version == "0.1"
        assert spec.project.name == "test-project"
        assert spec.providers.llm == {}
        assert spec.servers == []

    def test_full_valid_yaml(self):
        """Test parsing a complete YAML spec."""
        yaml_content = """
version: "0.1"
project:
  name: full-project

providers:
  llm:
    openai:
      api_key: env:OPENAI_API_KEY
      default_params:
        temperature: 0.7
        max_tokens: 2000

servers:
  - name: filesystem
    type: mcp
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

capabilities:
  - name: read_file
    server: filesystem
    method: read_file
    side_effect: read
    requires_approval: false
  - name: write_file
    server: filesystem
    method: write_file
    side_effect: write
    requires_approval: true

policies:
  - name: default
    budgets:
      max_cost_usd_per_run: 0.50
      max_capability_calls: 100
      timeout_seconds: 60

agents:
  - name: assistant
    provider: openai
    model:
      preference: gpt-4o-mini
      fallback: gpt-4o
    params:
      temperature: 0.5
    instructions: |
      You are a helpful assistant.
    allow:
      - read_file
      - write_file
    policy: default

workflows:
  - name: main
    entry: start
    steps:
      - id: start
        type: llm
        agent: assistant
        input:
          question: $input.question
        save_as: answer
        next: end
      - id: end
        type: end
"""
        spec = parse_yaml(yaml_content)

        assert spec.project.name == "full-project"
        assert "openai" in spec.providers.llm
        assert spec.providers.llm["openai"].api_key == "env:OPENAI_API_KEY"
        assert spec.providers.llm["openai"].default_params is not None
        assert spec.providers.llm["openai"].default_params.temperature == 0.7

        assert len(spec.servers) == 1
        assert spec.servers[0].name == "filesystem"

        assert len(spec.capabilities) == 2
        assert spec.capabilities[0].name == "read_file"
        assert spec.capabilities[0].side_effect == SideEffect.READ
        assert spec.capabilities[1].side_effect == SideEffect.WRITE
        assert spec.capabilities[1].requires_approval is True

        assert len(spec.policies) == 1
        assert spec.policies[0].budgets is not None
        assert spec.policies[0].budgets.timeout_seconds == 60

        assert len(spec.agents) == 1
        assert spec.agents[0].name == "assistant"
        assert spec.agents[0].model.preference == "gpt-4o-mini"
        assert spec.agents[0].model.fallback == "gpt-4o"
        assert len(spec.agents[0].allow) == 2

        assert len(spec.workflows) == 1
        assert spec.workflows[0].name == "main"
        assert spec.workflows[0].entry == "start"
        assert len(spec.workflows[0].steps) == 2

    def test_invalid_yaml_syntax(self):
        """Test that invalid YAML syntax raises ParseError."""
        invalid_yaml = """
version: "0.1"
project:
  name: test
    invalid_indent: value
"""
        with pytest.raises(ParseError) as exc_info:
            parse_yaml(invalid_yaml)
        assert "Invalid YAML syntax" in str(exc_info.value)

    def test_yaml_not_a_mapping(self):
        """Test that non-mapping YAML raises ParseError."""
        with pytest.raises(ParseError) as exc_info:
            parse_yaml("- item1\n- item2")
        assert "YAML root must be a mapping" in str(exc_info.value)

    def test_missing_required_field(self):
        """Test that missing required field raises ParseError."""
        yaml_content = """
version: "0.1"
# Missing project field
"""
        with pytest.raises(ParseError) as exc_info:
            parse_yaml(yaml_content)
        assert "Schema validation failed" in str(exc_info.value)
        assert "project" in str(exc_info.value)

    def test_invalid_step_type(self):
        """Test that invalid step type raises ParseError."""
        yaml_content = """
version: "0.1"
project:
  name: test

workflows:
  - name: main
    entry: start
    steps:
      - id: start
        type: invalid_type
"""
        with pytest.raises(ParseError) as exc_info:
            parse_yaml(yaml_content)
        assert "Schema validation failed" in str(exc_info.value)

    def test_condition_workflow(self):
        """Test parsing workflow with condition step."""
        yaml_content = """
version: "0.1"
project:
  name: conditional

providers:
  llm:
    openai:
      api_key: env:OPENAI_API_KEY

agents:
  - name: checker
    provider: openai
    model:
      preference: gpt-4
    instructions: Check things.

workflows:
  - name: check_flow
    entry: check
    steps:
      - id: check
        type: llm
        agent: checker
        save_as: result
        next: decide
      - id: decide
        type: condition
        condition: "$state.result == 'yes'"
        on_true: success
        on_false: failure
      - id: success
        type: end
      - id: failure
        type: end
"""
        spec = parse_yaml(yaml_content)
        workflow = spec.workflows[0]

        condition_step = next(s for s in workflow.steps if s.id == "decide")
        assert condition_step.type == StepType.CONDITION
        assert condition_step.condition == "$state.result == 'yes'"
        assert condition_step.on_true == "success"
        assert condition_step.on_false == "failure"

    def test_human_approval_workflow(self):
        """Test parsing workflow with human approval step."""
        yaml_content = """
version: "0.1"
project:
  name: approval-test

workflows:
  - name: with_approval
    entry: start
    steps:
      - id: start
        type: human_approval
        payload: $input.data
        on_approve: proceed
        on_reject: cancel
      - id: proceed
        type: end
      - id: cancel
        type: end
"""
        spec = parse_yaml(yaml_content)
        workflow = spec.workflows[0]

        approval_step = workflow.steps[0]
        assert approval_step.type == StepType.HUMAN_APPROVAL
        assert approval_step.payload == "$input.data"
        assert approval_step.on_approve == "proceed"
        assert approval_step.on_reject == "cancel"


class TestParseYamlFile:
    """Tests for parse_yaml_file function."""

    def test_parse_existing_file(self):
        """Test parsing an existing YAML file."""
        yaml_content = """
version: "0.1"
project:
  name: file-test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            spec = parse_yaml_file(f.name)
            assert spec.project.name == "file-test"

            # Cleanup
            Path(f.name).unlink()

    def test_file_not_found(self):
        """Test that non-existent file raises ParseError."""
        with pytest.raises(ParseError) as exc_info:
            parse_yaml_file("/nonexistent/path/file.yaml")
        assert "File not found" in str(exc_info.value)

    def test_parse_file_with_path_object(self):
        """Test parsing file using Path object."""
        yaml_content = """
version: "0.1"
project:
  name: path-test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            spec = parse_yaml_file(Path(f.name))
            assert spec.project.name == "path-test"

            # Cleanup
            Path(f.name).unlink()
