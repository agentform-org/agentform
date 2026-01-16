"""End-to-end tests for Agentform compiler."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler import (
    CompilationError,
    compile_agentform,
    compile_agentform_file,
    compile_file,
    validate_agentform_file,
    validate_file,
)


class TestCompileAgentform:
    """Test Agentform compilation from string."""

    def test_compiles_valid_agentform(self) -> None:
        """Test that valid Agentform compiles successfully."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "Answer clearly."
        }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "llm"
                agent = agent.assistant
                next = step.end
            }
            step "end" { type = "end" }
        }
        """
        compiled = compile_agentform(
            content,
            check_env=False,
            resolve_credentials=False,
            variables={"openai_api_key": "test-key"},
        )

        assert compiled.version == "0.1"
        assert compiled.project_name == "test"
        assert "openai" in compiled.providers
        assert "assistant" in compiled.agents
        assert "ask" in compiled.workflows

    def test_compile_error_on_invalid_agentform(self) -> None:
        """Test that invalid Agentform raises CompilationError."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.nonexistent
            instructions = "test"
        }
        """
        with pytest.raises(CompilationError) as exc_info:
            compile_agentform(content, check_env=False)

        assert "Unresolved reference" in str(exc_info.value)


class TestCompileAgentformFile:
    """Test Agentform compilation from file."""

    def test_compiles_agentform_file(self) -> None:
        """Test that Agentform file compiles successfully."""
        content = """
        agentform { version = "0.1" project = "file-test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".agentform", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                compiled = compile_agentform_file(
                    f.name,
                    check_env=False,
                    resolve_credentials=False,
                    variables={"openai_api_key": "test-key"},
                )
                assert compiled.project_name == "file-test"
            finally:
                Path(f.name).unlink()

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(CompilationError) as exc_info:
            compile_agentform_file("/nonexistent/path.agentform")

        assert "not found" in str(exc_info.value).lower()


class TestValidateAgentformFile:
    """Test Agentform file validation."""

    def test_validates_valid_file(self) -> None:
        """Test that valid file passes validation."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".agentform", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                result = validate_agentform_file(
                    f.name,
                    check_env=False,
                    variables={"openai_api_key": "test-key"},
                )
                assert result.is_valid
            finally:
                Path(f.name).unlink()

    def test_returns_errors_for_invalid_file(self) -> None:
        """Test that invalid file returns errors."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "assistant" {
            instructions = "test"
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".agentform", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                result = validate_agentform_file(f.name, check_env=False)
                assert not result.is_valid
                assert len(result.errors) > 0
            finally:
                Path(f.name).unlink()


class TestUnifiedCompileFile:
    """Test unified compile_file function with auto-detection."""

    def test_compiles_agentform_file(self) -> None:
        """Test that .agentform files are compiled correctly."""
        content = """
        agentform { version = "0.1" project = "agentform-test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".agentform", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                compiled = compile_file(
                    f.name,
                    check_env=False,
                    resolve_credentials=False,
                    variables={"openai_api_key": "test-key"},
                )
                assert compiled.project_name == "agentform-test"
            finally:
                Path(f.name).unlink()

    def test_compiles_yaml_file(self) -> None:
        """Test that .yaml files are rejected."""
        content = """
version: "0.1"
project:
  name: yaml-test
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                with pytest.raises(CompilationError) as exc_info:
                    compile_file(f.name, check_env=False, resolve_credentials=False)
                assert "Expected .agentform file" in str(exc_info.value) or ".yaml" in str(
                    exc_info.value
                )
            finally:
                Path(f.name).unlink()

    def test_unknown_extension_error(self) -> None:
        """Test error for unknown file extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("test")
            f.flush()

            try:
                with pytest.raises(CompilationError) as exc_info:
                    compile_file(f.name)

                assert "Expected .agentform file" in str(
                    exc_info.value
                ) or "Only .agentform files" in str(exc_info.value)
            finally:
                Path(f.name).unlink()


class TestUnifiedValidateFile:
    """Test unified validate_file function with auto-detection."""

    def test_validates_agentform_file(self) -> None:
        """Test that .agentform files are validated correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".agentform", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                result = validate_file(
                    f.name,
                    check_env=False,
                    variables={"openai_api_key": "test-key"},
                )
                assert result.is_valid
            finally:
                Path(f.name).unlink()

    def test_validates_yaml_file(self) -> None:
        """Test that .yaml files are rejected."""
        content = """
version: "0.1"
project:
  name: yaml-test
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                with pytest.raises(CompilationError) as exc_info:
                    validate_file(f.name, check_env=False)
                assert "Expected .agentform file" in str(exc_info.value) or ".yaml" in str(
                    exc_info.value
                )
            finally:
                Path(f.name).unlink()


class TestIROutput:
    """Test that compiled IR matches expected structure."""

    def test_ir_has_expected_fields(self) -> None:
        """Test that compiled IR has all expected fields."""
        content = """
        agentform { version = "0.2" project = "ir-test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
            params {
                temperature = 0.7
            }
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "Answer questions."
            policy = policy.default
        }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "llm"
                agent = agent.assistant
                input { question = input.question }
                output "answer" { from = result.text }
                next = step.end
            }
            step "end" { type = "end" }
        }
        """
        compiled = compile_agentform(
            content,
            check_env=False,
            resolve_credentials=False,
            variables={"openai_api_key": "env:OPENAI_API_KEY"},
        )

        # Check structure
        assert compiled.version == "0.2"
        assert compiled.project_name == "ir-test"

        # Providers
        assert "openai" in compiled.providers
        provider = compiled.providers["openai"]
        assert provider.api_key.env_var == "OPENAI_API_KEY"

        # Policies
        assert "default" in compiled.policies
        policy = compiled.policies["default"]
        assert policy.budgets.max_cost_usd_per_run == 0.50

        # Agents
        assert "assistant" in compiled.agents
        agent = compiled.agents["assistant"]
        assert agent.provider_name == "openai"
        assert agent.model_preference == "gpt-4o"
        assert agent.params.temperature == 0.7
        assert agent.policy_name == "default"

        # Workflows
        assert "ask" in compiled.workflows
        workflow = compiled.workflows["ask"]
        assert workflow.entry_step == "process"
        assert "process" in workflow.steps
        assert "end" in workflow.steps

        # Step details
        process_step = workflow.steps["process"]
        assert process_step.agent_name == "assistant"
        assert process_step.input_mapping is not None
        assert process_step.save_as == "answer"
        assert process_step.next_step == "end"
