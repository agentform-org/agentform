"""Tests for the main compiler module."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler.compiler import (
    CompilationError,
    compile_agentform,
    compile_agentform_file,
    validate_agentform_file,
)


class TestCompileSpec:
    """Tests for compile_agentform function."""

    def test_compile_minimal_spec(self):
        """Test compiling minimal valid spec."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}
"""
        ir = compile_agentform(agentform_content, check_env=False, resolve_credentials=False)
        assert ir.version == "0.1"
        assert ir.project_name == "test-project"

    def test_compile_full_spec(self, monkeypatch):
        """Test compiling a complete spec."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        agentform_content = """
agentform {
  version = "0.1"
  project = "full-test"
}

variable "openai_api_key" {
  default = "env:OPENAI_API_KEY"
  sensitive = true
}

provider "llm.openai" "default" {
  api_key = var.openai_api_key
  default_params {
    temperature = 0.7
  }
}

policy "default" {
  budgets { timeout_seconds = 60 }
}

model "gpt4" {
  provider = provider.llm.openai.default
  id = "gpt-4"
}

agent "assistant" {
  model = model.gpt4
  instructions = "You are helpful."
  policy = policy.default
}

workflow "main" {
  entry = step.start

  step "start" {
    type = "llm"
    agent = agent.assistant
    next = step.end
  }

  step "end" { type = "end" }
}
"""
        ir = compile_agentform(agentform_content, check_env=True, resolve_credentials=True)

        assert ir.project_name == "full-test"
        assert "openai" in ir.providers
        assert ir.providers["openai"].api_key.value == "sk-test"
        assert "default" in ir.policies
        assert "assistant" in ir.agents
        assert "main" in ir.workflows

    def test_compile_invalid_yaml(self):
        """Test that invalid Agentform raises CompilationError."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test"
  invalid = syntax error [
}
"""
        with pytest.raises(CompilationError) as exc_info:
            compile_agentform(agentform_content, check_env=False)
        assert "Parse error" in str(exc_info.value)

    def test_compile_validation_failure(self):
        """Test that validation errors raise CompilationError."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test"
}

agent "assistant" {
  model = model.nonexistent
  instructions = "Help."
}
"""
        with pytest.raises(CompilationError) as exc_info:
            compile_agentform(agentform_content, check_env=False)
        assert "Unresolved reference" in str(exc_info.value) or "Validation failed" in str(
            exc_info.value
        )
        if hasattr(exc_info.value, "validation_result") and exc_info.value.validation_result:
            assert not exc_info.value.validation_result.is_valid

    def test_compile_with_all_step_types(self, monkeypatch):
        """Test compiling spec with all step types."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        agentform_content = """
agentform {
  version = "0.1"
  project = "all-steps"
}

variable "openai_api_key" {
  default = "env:OPENAI_API_KEY"
  sensitive = true
}

provider "llm.openai" "default" {
  api_key = var.openai_api_key
}

server "fs" {
  type = "mcp"
  command = ["node", "fs-server"]
}

capability "read_file" {
  server = server.fs
  method = "readFile"
  side_effect = "read"
}

model "gpt4" {
  provider = provider.llm.openai.default
  id = "gpt-4"
}

agent "assistant" {
  model = model.gpt4
  instructions = "Help."
  allow = [capability.read_file]
}

workflow "complex" {
  entry = step.start

  step "start" {
    type = "llm"
    agent = agent.assistant
    save_as = "result"
    next = step.check
  }

  step "check" {
    type = "condition"
    condition = "$state.result == 'continue'"
    on_true = step.read
    on_false = step.approve
  }

  step "read" {
    type = "call"
    capability = capability.read_file
    args { path = "/tmp/test.txt" }
    next = step.end
  }

  step "approve" {
    type = "human_approval"
    payload = "$state.result"
    on_approve = step.end
    on_reject = step.end
  }

  step "end" { type = "end" }
}
"""
        ir = compile_agentform(agentform_content, check_env=False, resolve_credentials=False)

        workflow = ir.workflows["complex"]
        assert len(workflow.steps) == 5
        assert workflow.steps["start"].type.value == "llm"
        assert workflow.steps["check"].type.value == "condition"
        assert workflow.steps["read"].type.value == "call"
        assert workflow.steps["approve"].type.value == "human_approval"
        assert workflow.steps["end"].type.value == "end"


class TestCompileSpecFile:
    """Tests for compile_agentform_file function."""

    def test_compile_existing_file(self, monkeypatch):
        """Test compiling an existing Agentform file."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        agentform_content = """
agentform {
  version = "0.1"
  project = "file-test"
}

variable "openai_api_key" {
  default = "env:OPENAI_API_KEY"
  sensitive = true
}

provider "llm.openai" "default" {
  api_key = var.openai_api_key
}

model "gpt4" {
  provider = provider.llm.openai.default
  id = "gpt-4"
}

agent "assistant" {
  model = model.gpt4
  instructions = "Help."
}

workflow "main" {
  entry = step.start

  step "start" {
    type = "llm"
    agent = agent.assistant
    next = step.end
  }

  step "end" { type = "end" }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            f.flush()

            ir = compile_agentform_file(f.name, check_env=False, resolve_credentials=False)
            assert ir.project_name == "file-test"

            # Cleanup
            Path(f.name).unlink()

    def test_compile_nonexistent_file(self):
        """Test that non-existent file raises CompilationError."""
        with pytest.raises(CompilationError) as exc_info:
            compile_agentform_file("/nonexistent/path.agentform")
        assert "File not found" in str(exc_info.value) or "Parse error" in str(exc_info.value)

    def test_compile_file_with_path_object(self, monkeypatch):
        """Test compiling file using Path object."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "path-test"
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            f.flush()

            ir = compile_agentform_file(Path(f.name), check_env=False)
            assert ir.project_name == "path-test"

            Path(f.name).unlink()


class TestValidateSpecFile:
    """Tests for validate_agentform_file function."""

    def test_validate_valid_file(self):
        """Test validating a valid Agentform file."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "valid-test"
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            f.flush()

            result = validate_agentform_file(f.name, check_env=False)
            assert result.is_valid is True

            Path(f.name).unlink()

    def test_validate_invalid_file(self):
        """Test validating a file with validation errors."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "invalid-test"
}

agent "assistant" {
  model = model.nonexistent
  instructions = "Help."
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            f.flush()

            # Reference resolution errors raise CompilationError, not validation errors
            with pytest.raises(CompilationError) as exc_info:
                validate_agentform_file(f.name, check_env=False)
            assert "Unresolved reference" in str(
                exc_info.value
            ) or "Reference resolution failed" in str(exc_info.value)

            Path(f.name).unlink()

    def test_validate_nonexistent_file(self):
        """Test that non-existent file raises CompilationError."""
        with pytest.raises(CompilationError) as exc_info:
            validate_agentform_file("/nonexistent/path.agentform")
        assert "File not found" in str(exc_info.value) or "Parse error" in str(exc_info.value)
