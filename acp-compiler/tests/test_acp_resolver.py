"""Tests for ACP reference resolver."""

from acp_compiler.acp_parser import parse_acp
from acp_compiler.acp_resolver import resolve_references


class TestSymbolTableBuilding:
    """Test symbol table construction."""

    def test_builds_provider_symbols(self) -> None:
        """Test that provider symbols are registered."""
        content = """
        acp { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }
        variable "anthropic_api_key" { default = "env:ANTHROPIC_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        provider "llm.anthropic" "default" {
            api_key = var.anthropic_api_key
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert "provider.llm.openai.default" in result.symbols
        assert "provider.llm.anthropic.default" in result.symbols
        assert result.symbols["provider.llm.openai.default"].kind == "provider"

    def test_builds_model_symbols(self) -> None:
        """Test that model symbols are registered."""
        content = """
        acp { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        model "claude" {
            provider = provider.llm.anthropic.default
            id = "claude-3"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert "model.gpt4" in result.symbols
        assert "model.claude" in result.symbols
        assert result.symbols["model.gpt4"].kind == "model"

    def test_builds_agent_symbols(self) -> None:
        """Test that agent symbols are registered."""
        content = """
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert "agent.assistant" in result.symbols
        assert result.symbols["agent.assistant"].kind == "agent"

    def test_builds_workflow_and_step_symbols(self) -> None:
        """Test that workflow and step symbols are registered."""
        content = """
        acp { version = "0.1" project = "test" }

        workflow "ask" {
            entry = step.process
            step "process" { type = "llm" agent = agent.test next = step.end }
            step "end" { type = "end" }
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert "workflow.ask" in result.symbols
        assert "step.process" in result.symbols
        assert "step.end" in result.symbols
        assert result.symbols["step.process"].kind == "step"
        assert result.symbols["step.process"].parent == "ask"


class TestDuplicateDetection:
    """Test duplicate symbol detection."""

    def test_detects_duplicate_providers(self) -> None:
        """Test that duplicate providers are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        variable "key1" { default = "env:KEY1" }
        variable "key2" { default = "env:KEY2" }

        provider "llm.openai" "default" {
            api_key = var.key1
        }

        provider "llm.openai" "default" {
            api_key = var.key2
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Duplicate provider" in str(e) for e in result.errors)

    def test_detects_duplicate_models(self) -> None:
        """Test that duplicate models are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o-mini"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Duplicate model" in str(e) for e in result.errors)

    def test_detects_duplicate_steps_in_workflow(self) -> None:
        """Test that duplicate step IDs in a workflow are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        workflow "test" {
            entry = step.start
            step "start" { type = "llm" agent = agent.test }
            step "start" { type = "end" }
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Duplicate step" in str(e) for e in result.errors)


class TestReferenceResolution:
    """Test reference resolution."""

    def test_resolves_valid_model_provider_reference(self) -> None:
        """Test that valid model-to-provider references resolve."""
        content = """
        acp { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert result.is_valid

    def test_detects_unresolved_provider_reference(self) -> None:
        """Test that unresolved provider references are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.nonexistent.default
            id = "gpt-4o"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Unresolved reference" in str(e) for e in result.errors)

    def test_detects_unresolved_model_reference(self) -> None:
        """Test that unresolved model references are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.nonexistent
            instructions = "test"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Unresolved reference" in str(e) for e in result.errors)

    def test_detects_unresolved_step_reference(self) -> None:
        """Test that unresolved step references are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        workflow "test" {
            entry = step.nonexistent
            step "start" { type = "end" }
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("Unresolved reference" in str(e) for e in result.errors)

    def test_detects_wrong_reference_kind(self) -> None:
        """Test that references to wrong kinds are detected."""
        content = """
        acp { version = "0.1" project = "test" }

        policy "default" {
            budgets { timeout_seconds = 60 }
        }

        agent "assistant" {
            model = policy.default
            instructions = "test"
        }
        """
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert not result.is_valid
        assert any("expected model" in str(e) for e in result.errors)


class TestFullResolution:
    """Test complete resolution scenarios."""

    def test_resolves_complete_valid_spec(self) -> None:
        """Test that a complete valid spec resolves successfully."""
        content = """
        acp { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        policy "default" {
            budgets { timeout_seconds = 60 }
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        model "gpt4_mini" {
            provider = provider.llm.openai.default
            id = "gpt-4o-mini"
        }

        agent "assistant" {
            model = model.gpt4_mini
            fallback_models = [model.gpt4]
            instructions = "test"
            policy = policy.default
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
        acp_file = parse_acp(content)
        result = resolve_references(acp_file)

        assert result.is_valid
        assert len(result.errors) == 0
