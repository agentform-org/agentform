"""Tests for ACP validator."""

import pytest

from acp_compiler.acp_parser import parse_acp
from acp_compiler.acp_resolver import resolve_references
from acp_compiler.acp_validator import validate_acp


class TestACPBlockValidation:
    """Test validation of the acp metadata block."""

    def test_missing_acp_block(self) -> None:
        """Test error when acp block is missing."""
        content = '''
        provider "llm.openai" "default" {
            api_key = env("KEY")
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("Missing required 'acp' block" in e.message for e in result.errors)

    def test_missing_version(self) -> None:
        """Test error when version is missing."""
        content = '''
        acp { project = "test" }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("version" in e.path for e in result.errors)

    def test_missing_project(self) -> None:
        """Test error when project is missing."""
        content = '''
        acp { version = "0.1" }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("project" in e.path for e in result.errors)


class TestProviderValidation:
    """Test validation of provider blocks."""

    def test_missing_api_key(self) -> None:
        """Test error when api_key is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        provider "llm.openai" "default" {
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("api_key" in e.path for e in result.errors)

    def test_api_key_must_use_env(self) -> None:
        """Test error when api_key doesn't use env()."""
        content = '''
        acp { version = "0.1" project = "test" }

        provider "llm.openai" "default" {
            api_key = "hardcoded-key"
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("env()" in e.message for e in result.errors)


class TestModelValidation:
    """Test validation of model blocks."""

    def test_missing_provider(self) -> None:
        """Test error when provider is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        model "gpt4" {
            id = "gpt-4o"
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("provider" in e.path for e in result.errors)

    def test_missing_id(self) -> None:
        """Test error when id is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.openai.default
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("id" in e.path for e in result.errors)


class TestAgentValidation:
    """Test validation of agent blocks."""

    def test_missing_model(self) -> None:
        """Test error when model is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            instructions = "test"
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("model" in e.path for e in result.errors)

    def test_missing_instructions(self) -> None:
        """Test error when instructions is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.gpt4
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("instructions" in e.path for e in result.errors)


class TestWorkflowValidation:
    """Test validation of workflow blocks."""

    def test_missing_entry(self) -> None:
        """Test error when entry is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            step "process" { type = "end" }
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("entry" in e.path for e in result.errors)

    def test_empty_workflow(self) -> None:
        """Test error when workflow has no steps."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("at least one step" in e.message for e in result.errors)


class TestStepValidation:
    """Test validation of workflow steps."""

    def test_missing_step_type(self) -> None:
        """Test error when step type is missing."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                agent = agent.test
            }
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("type" in e.path for e in result.errors)

    def test_invalid_step_type(self) -> None:
        """Test error when step type is invalid."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "invalid_type"
            }
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("Invalid step type" in e.message for e in result.errors)

    def test_llm_step_requires_agent(self) -> None:
        """Test error when LLM step has no agent."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" {
                type = "llm"
            }
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("agent" in e.path for e in result.errors)

    def test_call_step_requires_capability(self) -> None:
        """Test error when call step has no capability."""
        content = '''
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.call
            step "call" {
                type = "call"
            }
        }
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("capability" in e.path for e in result.errors)

    def test_condition_step_requires_condition(self) -> None:
        """Test error when condition step has no condition."""
        content = '''
        acp { version = "0.1" project = "test" }

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
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert not result.is_valid
        assert any("condition" in e.path for e in result.errors)


class TestValidSpec:
    """Test validation of complete valid specifications."""

    def test_valid_minimal_spec(self) -> None:
        """Test that a minimal valid spec passes."""
        content = '''
        acp { version = "0.1" project = "test" }

        provider "llm.openai" "default" {
            api_key = env("OPENAI_API_KEY")
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
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_valid_complete_spec(self) -> None:
        """Test that a complete valid spec passes."""
        content = '''
        acp { version = "0.2" project = "complete-test" }

        provider "llm.openai" "default" {
            api_key = env("OPENAI_API_KEY")
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
        '''
        acp_file = parse_acp(content)
        resolution = resolve_references(acp_file)
        result = validate_acp(acp_file, resolution, check_env=False)

        assert result.is_valid
        assert len(result.errors) == 0

