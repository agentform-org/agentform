"""Tests for Agentform validator."""

from agentform_compiler.agentform_parser import parse_agentform
from agentform_compiler.agentform_resolver import resolve_references
from agentform_compiler.agentform_validator import validate_agentform


class TestAgentformBlockValidation:
    """Test validation of the agentform metadata block."""

    def test_missing_agentform_block(self) -> None:
        """Test error when agentform block is missing."""
        content = """
        variable "api_key" { type = string }
        provider "llm.openai" "default" {
            api_key = var.api_key
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("Missing required 'agentform' block" in e.message for e in result.errors)

    def test_missing_version(self) -> None:
        """Test error when version is missing."""
        content = """
        agentform { project = "test" }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("version" in e.path for e in result.errors)

    def test_missing_project(self) -> None:
        """Test error when project is missing."""
        content = """
        agentform { version = "0.1" }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("project" in e.path for e in result.errors)


class TestProviderValidation:
    """Test validation of provider blocks."""

    def test_missing_api_key(self) -> None:
        """Test error when api_key is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        provider "llm.openai" "default" {
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("api_key" in e.path for e in result.errors)

    def test_api_key_must_use_var_ref(self) -> None:
        """Test error when api_key doesn't use var reference."""
        content = """
        agentform { version = "0.1" project = "test" }

        provider "llm.openai" "default" {
            api_key = "hardcoded-key"
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("variable reference" in e.message for e in result.errors)


class TestModelValidation:
    """Test validation of model blocks."""

    def test_missing_provider(self) -> None:
        """Test error when provider is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        model "gpt4" {
            id = "gpt-4o"
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("provider" in e.path for e in result.errors)

    def test_missing_id(self) -> None:
        """Test error when id is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.openai.default
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("id" in e.path for e in result.errors)


class TestAgentValidation:
    """Test validation of agent blocks."""

    def test_missing_model(self) -> None:
        """Test error when model is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "assistant" {
            instructions = "test"
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("model" in e.path for e in result.errors)

    def test_missing_instructions(self) -> None:
        """Test error when instructions is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.gpt4
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("instructions" in e.path for e in result.errors)


class TestWorkflowValidation:
    """Test validation of workflow blocks."""

    def test_missing_entry(self) -> None:
        """Test error when entry is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            step "process" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("entry" in e.path for e in result.errors)

    def test_empty_workflow(self) -> None:
        """Test error when workflow has no steps."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("at least one step" in e.message for e in result.errors)


class TestStepValidation:
    """Test validation of workflow steps."""

    def test_missing_step_type(self) -> None:
        """Test error when step type is missing."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                agent = agent.test
            }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("type" in e.path for e in result.errors)

    def test_invalid_step_type(self) -> None:
        """Test error when step type is invalid."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "invalid_type"
            }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("Invalid step type" in e.message for e in result.errors)

    def test_llm_step_requires_agent(self) -> None:
        """Test error when LLM step has no agent."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "llm"
            }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("agent" in e.path for e in result.errors)

    def test_call_step_requires_capability(self) -> None:
        """Test error when call step has no capability."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.call
            step "call" {
                type = "call"
            }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("capability" in e.path for e in result.errors)

    def test_condition_step_requires_condition(self) -> None:
        """Test error when condition step has no condition."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.route
            step "route" {
                type = "condition"
                on_true = step.a
                on_false = step.b
            }
            step "a" { type = "end" }
            step "b" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("condition" in e.path for e in result.errors)


class TestValidSpec:
    """Test validation of complete valid specifications."""

    def test_valid_minimal_spec(self) -> None:
        """Test that a minimal valid spec passes."""
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
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_complete_spec(self) -> None:
        """Test that a complete valid spec passes."""
        content = """
        agentform { version = "0.2" project = "complete-test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        server "filesystem" {
            type = "mcp"
            transport = "stdio"
            command = ["npx", "server"]
        }

        capability "read_file" {
            server = server.filesystem
            method = "read_file"
            side_effect = "read"
            requires_approval = false
        }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
            budgets { timeout_seconds = 60 }
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
            params {
                temperature = 0.7
                max_tokens = 2000
            }
        }

        agent "assistant" {
            model = model.gpt4
            instructions = "Answer clearly."
            policy = policy.default
            allow = [capability.read_file]
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
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert result.is_valid
        assert len(result.errors) == 0


class TestVariableValidation:
    """Test validation of variable blocks."""

    def test_invalid_variable_type(self) -> None:
        """Test error when variable type is invalid."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "test" {
            type = invalid_type
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        assert not result.is_valid
        assert any("Invalid variable type" in e.message for e in result.errors)

    def test_warning_for_required_variable(self) -> None:
        """Test warning when variable has no default and is not sensitive."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "test" {
            type = string
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        # Should have a warning about no default value
        assert any("no default value" in w.message for w in result.warnings)

    def test_valid_variable_with_default(self) -> None:
        """Test that a variable with default passes."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "temperature" {
            type = number
            default = 0.7
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        result = validate_agentform(agentform_file, resolution)

        # Should pass without errors
        assert result.is_valid
