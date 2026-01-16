"""Tests for the validate command."""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from agentform_cli.main import app

runner = CliRunner()


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_nonexistent_file(self):
        """Test validating a file that doesn't exist."""
        result = runner.invoke(app, ["validate", "nonexistent.agentform"])
        assert result.exit_code == 1
        assert "Path not found" in result.stdout

    def test_validate_invalid_yaml(self):
        """Test validating invalid Agentform."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write("invalid agentform content [")
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["validate", str(temp_path)])
            assert result.exit_code == 1
            assert "Parse error" in result.stdout or "error" in result.stdout.lower()
        finally:
            temp_path.unlink()

    def test_validate_valid_minimal_spec(self):
        """Test validating a valid minimal spec."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["validate", str(temp_path)])
            assert result.exit_code == 0
            assert "Validation passed" in result.stdout
        finally:
            temp_path.unlink()

    def test_validate_with_check_env(self):
        """Test validate with environment check enabled."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["validate", str(temp_path), "--check-env"])
            # Should pass for minimal spec without env vars
            assert result.exit_code == 0
        finally:
            temp_path.unlink()

    def test_validate_no_check_env(self):
        """Test validate with environment check disabled."""
        agentform_content = """
agentform {
  version = "0.1"
  project = "test-project"
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["validate", str(temp_path), "--no-check-env"])
            assert result.exit_code == 0
            assert "Validation passed" in result.stdout
        finally:
            temp_path.unlink()

    def test_validate_full_spec(self, monkeypatch):
        """Test validating a full spec with providers."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        agentform_content = """
agentform {
  version = "0.1"
  project = "full-test"
}

variable "openai_api_key" {
  type = string
  default = "sk-test"
  sensitive = true
}

provider "llm.openai" "default" {
  api_key = var.openai_api_key
}

policy "default" {
  budgets { timeout_seconds = 60 }
}

model "gpt4o_mini" {
  provider = provider.llm.openai.default
  id = "gpt-4o-mini"
}

agent "assistant" {
  model = model.gpt4o_mini
  instructions = "You are helpful"
  policy = policy.default
}

workflow "ask" {
  entry = step.process

  step "process" {
    type = "llm"
    agent = agent.assistant
    input { question = input.question }
    next = step.end
  }

  step "end" { type = "end" }
}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".agentform", delete=False) as f:
            f.write(agentform_content)
            temp_path = Path(f.name)

        try:
            result = runner.invoke(app, ["validate", str(temp_path), "--no-check-env"])
            assert result.exit_code == 0
            assert "Validation passed" in result.stdout
            assert "Summary" in result.stdout or "Workflows" in result.stdout
        finally:
            temp_path.unlink()
