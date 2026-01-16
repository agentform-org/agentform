"""Tests for Agentform normalizer."""

from agentform_compiler.agentform_normalizer import normalize_agentform
from agentform_compiler.agentform_parser import parse_agentform
from agentform_compiler.agentform_resolver import resolve_references
from agentform_schema.models import SideEffect, StepType


class TestProviderNormalization:
    """Test provider normalization."""

    def test_normalizes_provider(self) -> None:
        """Test that providers are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert "openai" in spec.providers.llm
        provider = spec.providers.llm["openai"]
        assert provider.api_key == "env:OPENAI_API_KEY"

    def test_normalizes_provider_with_custom_name(self) -> None:
        """Test that non-default provider names include the name."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_prod_key" { default = "env:OPENAI_PROD_KEY" }

        provider "llm.openai" "production" {
            api_key = var.openai_prod_key
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert "openai_production" in spec.providers.llm


class TestModelNormalization:
    """Test model normalization."""

    def test_model_info_embedded_in_agent(self) -> None:
        """Test that model info is embedded in agent config."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
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
            instructions = "test"
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert len(spec.agents) == 1
        agent = spec.agents[0]
        assert agent.provider == "openai"
        assert agent.model.preference == "gpt-4o"
        assert agent.params is not None
        assert agent.params.temperature == 0.7

    def test_fallback_model_normalized(self) -> None:
        """Test that fallback models are normalized."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
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
        }

        workflow "ask" {
            entry = step.end
            step "end" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        agent = spec.agents[0]
        assert agent.model.preference == "gpt-4o-mini"
        assert agent.model.fallback == "gpt-4o"


class TestServerNormalization:
    """Test server normalization."""

    def test_normalizes_server(self) -> None:
        """Test that servers are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        server "filesystem" {
            type = "mcp"
            transport = "stdio"
            command = ["npx", "server", "/path"]
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert len(spec.servers) == 1
        server = spec.servers[0]
        assert server.name == "filesystem"
        assert server.type == "mcp"
        assert server.command == ["npx", "server", "/path"]


class TestCapabilityNormalization:
    """Test capability normalization."""

    def test_normalizes_capability(self) -> None:
        """Test that capabilities are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        server "fs" {
            command = ["npx", "server"]
        }

        capability "write_file" {
            server = server.fs
            method = "write_file"
            side_effect = "write"
            requires_approval = true
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert len(spec.capabilities) == 1
        cap = spec.capabilities[0]
        assert cap.name == "write_file"
        assert cap.server == "fs"
        assert cap.method == "write_file"
        assert cap.side_effect == SideEffect.WRITE
        assert cap.requires_approval is True


class TestPolicyNormalization:
    """Test policy normalization."""

    def test_normalizes_policy(self) -> None:
        """Test that policies are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
            budgets { timeout_seconds = 60 }
            budgets { max_capability_calls = 10 }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert len(spec.policies) == 1
        policy = spec.policies[0]
        assert policy.name == "default"
        assert policy.budgets is not None
        assert policy.budgets.max_cost_usd_per_run == 0.50
        assert policy.budgets.timeout_seconds == 60
        assert policy.budgets.max_capability_calls == 10


class TestWorkflowNormalization:
    """Test workflow normalization."""

    def test_normalizes_workflow(self) -> None:
        """Test that workflows are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "api_key" { default = "env:KEY" }

        provider "llm.openai" "default" {
            api_key = var.api_key
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
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        assert len(spec.workflows) == 1
        workflow = spec.workflows[0]
        assert workflow.name == "ask"
        assert workflow.entry == "process"
        assert len(workflow.steps) == 2

        # Check process step
        process_step = next(s for s in workflow.steps if s.id == "process")
        assert process_step.type == StepType.LLM
        assert process_step.agent == "assistant"
        assert process_step.input is not None
        assert process_step.input["question"] == "$input.question"
        assert process_step.save_as == "answer"
        assert process_step.next == "end"

        # Check end step
        end_step = next(s for s in workflow.steps if s.id == "end")
        assert end_step.type == StepType.END


class TestStepTypeNormalization:
    """Test step type normalization."""

    def test_normalizes_call_step(self) -> None:
        """Test that call steps are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        server "fs" { command = ["npx", "server"] }

        capability "read_file" {
            server = server.fs
            method = "read"
        }

        workflow "read" {
            entry = step.call

            step "call" {
                type = "call"
                capability = capability.read_file
                args { path = input.file_path }
                output "content" { from = result.data }
                next = step.end
            }

            step "end" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        workflow = spec.workflows[0]
        call_step = next(s for s in workflow.steps if s.id == "call")
        assert call_step.type == StepType.CALL
        assert call_step.capability == "read_file"
        assert call_step.args is not None
        assert call_step.args["path"] == "$input.file_path"

    def test_normalizes_condition_step(self) -> None:
        """Test that condition steps are normalized correctly."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "route" {
            entry = step.check

            step "check" {
                type = "condition"
                condition = "state.value > 0"
                on_true = step.yes
                on_false = step.no
            }

            step "yes" { type = "end" }
            step "no" { type = "end" }
        }
        """
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        workflow = spec.workflows[0]
        check_step = next(s for s in workflow.steps if s.id == "check")
        assert check_step.type == StepType.CONDITION
        assert check_step.condition == "state.value > 0"
        assert check_step.on_true == "yes"
        assert check_step.on_false == "no"


class TestFullNormalization:
    """Test complete normalization scenarios."""

    def test_normalizes_complete_spec(self) -> None:
        """Test that a complete spec normalizes correctly."""
        content = """
        agentform { version = "0.2" project = "complete-test" }

        variable "openai_api_key" { default = "env:OPENAI_API_KEY" }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
            budgets { timeout_seconds = 60 }
        }

        model "gpt4_mini" {
            provider = provider.llm.openai.default
            id = "gpt-4o-mini"
            params {
                temperature = 0.7
                max_tokens = 2000
            }
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4_mini
            fallback_models = [model.gpt4]
            instructions = "Answer clearly."
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
        agentform_file = parse_agentform(content)
        resolution = resolve_references(agentform_file)
        spec = normalize_agentform(agentform_file, resolution)

        # Check version and project
        assert spec.version == "0.2"
        assert spec.project.name == "complete-test"

        # Check providers
        assert "openai" in spec.providers.llm

        # Check policies
        assert len(spec.policies) == 1

        # Check agents
        assert len(spec.agents) == 1
        agent = spec.agents[0]
        assert agent.name == "assistant"
        assert agent.model.preference == "gpt-4o-mini"
        assert agent.model.fallback == "gpt-4o"
        assert agent.policy == "default"

        # Check workflows
        assert len(spec.workflows) == 1
        workflow = spec.workflows[0]
        assert workflow.name == "ask"
        assert len(workflow.steps) == 2
