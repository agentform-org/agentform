"""End-to-end tests for ACP compiler."""

import tempfile
from pathlib import Path

import pytest

from acp_compiler import (
    CompilationError,
    compile_acp,
    compile_acp_file,
    compile_file,
    validate_acp_file,
    validate_file,
)


class TestCompileACP:
    """Test ACP compilation from string."""

    def test_compiles_valid_acp(self) -> None:
        """Test that valid ACP compiles successfully."""
        content = """
        acp { version = "0.1" project = "test" }

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
        compiled = compile_acp(
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

    def test_compile_error_on_invalid_acp(self) -> None:
        """Test that invalid ACP raises CompilationError."""
        content = """
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.nonexistent
            instructions = "test"
        }
        """
        with pytest.raises(CompilationError) as exc_info:
            compile_acp(content, check_env=False)

        assert "Unresolved reference" in str(exc_info.value)


class TestCompileACPFile:
    """Test ACP compilation from file."""

    def test_compiles_acp_file(self) -> None:
        """Test that ACP file compiles successfully."""
        content = """
        acp { version = "0.1" project = "file-test" }

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
        with tempfile.NamedTemporaryFile(suffix=".acp", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                compiled = compile_acp_file(
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
            compile_acp_file("/nonexistent/path.acp")

        assert "not found" in str(exc_info.value).lower()


class TestValidateACPFile:
    """Test ACP file validation."""

    def test_validates_valid_file(self) -> None:
        """Test that valid file passes validation."""
        content = """
        acp { version = "0.1" project = "test" }

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
        with tempfile.NamedTemporaryFile(suffix=".acp", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                result = validate_acp_file(
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
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            instructions = "test"
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".acp", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                result = validate_acp_file(f.name, check_env=False)
                assert not result.is_valid
                assert len(result.errors) > 0
            finally:
                Path(f.name).unlink()


class TestUnifiedCompileFile:
    """Test unified compile_file function with auto-detection."""

    def test_compiles_acp_file(self) -> None:
        """Test that .acp files are compiled correctly."""
        content = """
        acp { version = "0.1" project = "acp-test" }

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
        with tempfile.NamedTemporaryFile(suffix=".acp", delete=False, mode="w") as f:
            f.write(content)
            f.flush()

            try:
                compiled = compile_file(
                    f.name,
                    check_env=False,
                    resolve_credentials=False,
                    variables={"openai_api_key": "test-key"},
                )
                assert compiled.project_name == "acp-test"
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
                assert "Expected .acp file" in str(exc_info.value) or ".yaml" in str(exc_info.value)
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

                assert "Expected .acp file" in str(exc_info.value) or "Only .acp files" in str(
                    exc_info.value
                )
            finally:
                Path(f.name).unlink()


class TestUnifiedValidateFile:
    """Test unified validate_file function with auto-detection."""

    def test_validates_acp_file(self) -> None:
        """Test that .acp files are validated correctly."""
        content = """
        acp { version = "0.1" project = "test" }

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
        with tempfile.NamedTemporaryFile(suffix=".acp", delete=False, mode="w") as f:
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
                assert "Expected .acp file" in str(exc_info.value) or ".yaml" in str(exc_info.value)
            finally:
                Path(f.name).unlink()


class TestIROutput:
    """Test that compiled IR matches expected structure."""

    def test_ir_has_expected_fields(self) -> None:
        """Test that compiled IR has all expected fields."""
        content = """
        acp { version = "0.2" project = "ir-test" }

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
        compiled = compile_acp(
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
