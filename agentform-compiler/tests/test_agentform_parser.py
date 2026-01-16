"""Tests for Agentform parser."""

import pytest

from agentform_compiler.agentform_ast import (
    AgentformFile,
    AndExpr,
    ComparisonExpr,
    ConditionalExpr,
    NotExpr,
    OrExpr,
    Reference,
    StateRef,
    VarRef,
)
from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform


class TestBasicParsing:
    """Test basic parsing functionality."""

    def test_parse_empty_agentform_block(self) -> None:
        """Test parsing an empty agentform block."""
        content = 'agentform { version = "0.1" project = "test" }'
        result = parse_agentform(content)

        assert isinstance(result, AgentformFile)
        assert result.agentform is not None
        assert result.agentform.version == "0.1"
        assert result.agentform.project == "test"

    def test_parse_provider_block(self) -> None:
        """Test parsing a provider block."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }
        """
        result = parse_agentform(content)

        assert len(result.providers) == 1
        provider = result.providers[0]
        assert provider.provider_type == "llm.openai"
        assert provider.name == "default"
        assert provider.full_name == "llm.openai.default"

        api_key = provider.get_attribute("api_key")
        assert isinstance(api_key, VarRef)
        assert api_key.var_name == "openai_api_key"

    def test_parse_model_block(self) -> None:
        """Test parsing a model block."""
        content = """
        agentform { version = "0.1" project = "test" }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
            params {
                temperature = 0.7
                max_tokens = 2000
            }
        }
        """
        result = parse_agentform(content)

        assert len(result.models) == 1
        model = result.models[0]
        assert model.name == "gpt4"

        provider_ref = model.get_attribute("provider")
        assert isinstance(provider_ref, Reference)
        assert provider_ref.path == "provider.llm.openai.default"

        model_id = model.get_attribute("id")
        assert model_id == "gpt-4o"

        params_block = model.get_params_block()
        assert params_block is not None
        assert params_block.get_attribute("temperature") == 0.7
        assert params_block.get_attribute("max_tokens") == 2000

    def test_parse_agent_block(self) -> None:
        """Test parsing an agent block."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "assistant" {
            model = model.gpt4
            fallback_models = [model.gpt4_mini]
            instructions = "Answer clearly."
            policy = policy.default
        }
        """
        result = parse_agentform(content)

        assert len(result.agents) == 1
        agent = result.agents[0]
        assert agent.name == "assistant"

        model_ref = agent.get_attribute("model")
        assert isinstance(model_ref, Reference)
        assert model_ref.path == "model.gpt4"

        fallback = agent.get_attribute("fallback_models")
        assert isinstance(fallback, list)
        assert len(fallback) == 1
        assert isinstance(fallback[0], Reference)
        assert fallback[0].path == "model.gpt4_mini"

        instructions = agent.get_attribute("instructions")
        assert instructions == "Answer clearly."

    def test_parse_workflow_block(self) -> None:
        """Test parsing a workflow block with steps."""
        content = """
        agentform { version = "0.1" project = "test" }

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
        result = parse_agentform(content)

        assert len(result.workflows) == 1
        workflow = result.workflows[0]
        assert workflow.name == "ask"

        entry_ref = workflow.get_attribute("entry")
        assert isinstance(entry_ref, Reference)
        assert entry_ref.path == "step.process"

        assert len(workflow.steps) == 2

        # Check process step
        process_step = workflow.steps[0]
        assert process_step.step_id == "process"
        assert process_step.get_attribute("type") == "llm"

        agent_ref = process_step.get_attribute("agent")
        assert isinstance(agent_ref, Reference)
        assert agent_ref.path == "agent.assistant"

        input_block = process_step.get_input_block()
        assert input_block is not None

        output_blocks = process_step.get_output_blocks()
        assert len(output_blocks) == 1
        assert output_blocks[0].label == "answer"

        # Check end step
        end_step = workflow.steps[1]
        assert end_step.step_id == "end"
        assert end_step.get_attribute("type") == "end"


class TestValueParsing:
    """Test parsing of different value types."""

    def test_parse_string_value(self) -> None:
        """Test parsing string values."""
        content = """
        agentform { version = "0.1" project = "test project" }
        """
        result = parse_agentform(content)
        assert result.agentform is not None
        assert result.agentform.project == "test project"

    def test_parse_number_values(self) -> None:
        """Test parsing integer and float values."""
        content = """
        agentform { version = "0.1" project = "test" }

        model "test" {
            provider = provider.llm.openai.default
            id = "test"
            params {
                temperature = 0.7
                max_tokens = 2000
                top_p = 0.95
            }
        }
        """
        result = parse_agentform(content)
        params = result.models[0].get_params_block()
        assert params is not None
        assert params.get_attribute("temperature") == 0.7
        assert params.get_attribute("max_tokens") == 2000
        assert params.get_attribute("top_p") == 0.95

    def test_parse_boolean_values(self) -> None:
        """Test parsing boolean values."""
        content = """
        agentform { version = "0.1" project = "test" }

        capability "write_file" {
            server = server.filesystem
            method = "write"
            side_effect = "write"
            requires_approval = true
        }
        """
        result = parse_agentform(content)
        cap = result.capabilities[0]
        assert cap.get_attribute("requires_approval") is True

    def test_parse_array_values(self) -> None:
        """Test parsing array values."""
        content = """
        agentform { version = "0.1" project = "test" }

        server "fs" {
            command = ["npx", "server", "/path"]
            transport = "stdio"
        }
        """
        result = parse_agentform(content)
        server = result.servers[0]
        command = server.get_attribute("command")
        assert isinstance(command, list)
        assert command == ["npx", "server", "/path"]

    def test_parse_reference_values(self) -> None:
        """Test parsing reference values."""
        content = """
        agentform { version = "0.1" project = "test" }

        agent "test" {
            model = model.gpt4
            policy = policy.default
            instructions = "test"
        }
        """
        result = parse_agentform(content)
        agent = result.agents[0]

        model_ref = agent.get_attribute("model")
        assert isinstance(model_ref, Reference)
        assert model_ref.parts == ["model", "gpt4"]

    def test_parse_var_ref(self) -> None:
        """Test parsing variable references."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "anthropic_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.anthropic" "default" {
            api_key = var.anthropic_api_key
        }
        """
        result = parse_agentform(content)
        provider = result.providers[0]
        api_key = provider.get_attribute("api_key")
        assert isinstance(api_key, VarRef)
        assert api_key.var_name == "anthropic_api_key"


class TestNestedBlocks:
    """Test parsing of nested blocks."""

    def test_parse_unlabeled_nested_block(self) -> None:
        """Test parsing nested blocks without labels."""
        content = """
        agentform { version = "0.1" project = "test" }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
            budgets { timeout_seconds = 60 }
        }
        """
        result = parse_agentform(content)
        policy = result.policies[0]
        budget_blocks = policy.get_budgets_blocks()
        assert len(budget_blocks) == 2

    def test_parse_labeled_nested_block(self) -> None:
        """Test parsing nested blocks with labels."""
        content = """
        agentform { version = "0.1" project = "test" }

        workflow "test" {
            entry = step.start
            step "start" {
                type = "llm"
                agent = agent.test
                output "result" { from = result.text }
                next = step.end
            }
            step "end" { type = "end" }
        }
        """
        result = parse_agentform(content)
        step = result.workflows[0].steps[0]
        outputs = step.get_output_blocks()
        assert len(outputs) == 1
        assert outputs[0].label == "result"


class TestComments:
    """Test parsing with comments."""

    def test_line_comments(self) -> None:
        """Test that line comments are ignored."""
        content = """
        // This is a comment
        agentform {
            version = "0.1"  // inline comment
            project = "test"
        }
        // Another comment
        """
        result = parse_agentform(content)
        assert result.agentform is not None
        assert result.agentform.version == "0.1"

    def test_block_comments(self) -> None:
        """Test that block comments are ignored."""
        content = """
        /* Block comment */
        agentform {
            version = "0.1"
            /* multi
               line
               comment */
            project = "test"
        }
        """
        result = parse_agentform(content)
        assert result.agentform is not None
        assert result.agentform.project == "test"


class TestParseErrors:
    """Test parse error handling."""

    def test_missing_closing_brace(self) -> None:
        """Test error on missing closing brace."""
        content = """
        agentform { version = "0.1" project = "test"
        """
        with pytest.raises(AgentformParseError):
            parse_agentform(content)

    def test_invalid_token(self) -> None:
        """Test error on invalid token."""
        content = """
        agentform { version = @invalid }
        """
        with pytest.raises(AgentformParseError):
            parse_agentform(content)

    def test_missing_equals(self) -> None:
        """Test error on missing equals sign."""
        content = """
        agentform { version "0.1" }
        """
        with pytest.raises(AgentformParseError):
            parse_agentform(content)


class TestFullExample:
    """Test parsing a complete example."""

    def test_parse_full_example(self) -> None:
        """Test parsing the example from the PRD."""
        content = """
        agentform {
            version = "0.2"
            project = "models-demo"
        }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.openai_api_key
        }

        policy "default" {
            budgets { max_cost_usd_per_run = 0.50 }
            budgets { timeout_seconds = 60 }
        }

        model "openai_gpt4o_mini" {
            provider = provider.llm.openai.default
            id = "gpt-4o-mini"
            params {
                temperature = 0.7
                max_tokens = 2000
            }
        }

        model "openai_gpt4o" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
            params {
                temperature = 0.4
                max_tokens = 2500
            }
        }

        agent "assistant" {
            model = model.openai_gpt4o_mini
            fallback_models = [model.openai_gpt4o]
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
        result = parse_agentform(content)

        assert result.agentform is not None
        assert result.agentform.version == "0.2"
        assert result.agentform.project == "models-demo"

        assert len(result.variables) == 1
        assert len(result.providers) == 1
        assert len(result.policies) == 1
        assert len(result.models) == 2
        assert len(result.agents) == 1
        assert len(result.workflows) == 1


class TestVariableParsing:
    """Test parsing of variable blocks."""

    def test_parse_variable_block(self) -> None:
        """Test parsing a variable block."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "api_key" {
            type = string
            description = "The API key"
            sensitive = true
        }
        """
        result = parse_agentform(content)

        assert len(result.variables) == 1
        var = result.variables[0]
        assert var.name == "api_key"
        assert var.var_type == "string"
        assert var.description == "The API key"
        assert var.sensitive is True

    def test_parse_variable_with_default(self) -> None:
        """Test parsing a variable with default value."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "temperature" {
            type = number
            default = 0.7
        }
        """
        result = parse_agentform(content)

        assert len(result.variables) == 1
        var = result.variables[0]
        assert var.name == "temperature"
        assert var.var_type == "number"
        assert var.default == 0.7
        assert var.sensitive is False

    def test_parse_multiple_variables(self) -> None:
        """Test parsing multiple variable blocks."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "openai_api_key" {
            type = string
            sensitive = true
        }

        variable "anthropic_api_key" {
            type = string
            sensitive = true
        }

        variable "max_tokens" {
            type = number
            default = 2000
        }
        """
        result = parse_agentform(content)

        assert len(result.variables) == 3
        assert result.variables[0].name == "openai_api_key"
        assert result.variables[1].name == "anthropic_api_key"
        assert result.variables[2].name == "max_tokens"

    def test_parse_var_ref(self) -> None:
        """Test parsing variable references in attributes."""
        content = """
        agentform { version = "0.1" project = "test" }

        variable "api_key" {
            type = string
            sensitive = true
        }

        provider "llm.openai" "default" {
            api_key = var.api_key
        }
        """
        result = parse_agentform(content)

        provider = result.providers[0]
        api_key = provider.get_attribute("api_key")
        assert isinstance(api_key, VarRef)
        assert api_key.var_name == "api_key"


class TestConditionalExpressions:
    """Test parsing conditional expressions (Terraform-style)."""

    def test_parse_simple_conditional(self) -> None:
        """Test parsing a simple conditional expression."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            id = $input.use_mini ? "gpt-4o-mini" : "gpt-4o"
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        model_id = model.get_attribute("id")

        assert isinstance(model_id, ConditionalExpr)
        assert isinstance(model_id.condition, StateRef)
        assert model_id.condition.path == "$input.use_mini"
        assert model_id.true_value == "gpt-4o-mini"
        assert model_id.false_value == "gpt-4o"

    def test_parse_conditional_with_comparison(self) -> None:
        """Test parsing conditional with comparison in condition."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            id = $input.env == "prod" ? "gpt-4o" : "gpt-4o-mini"
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        model_id = model.get_attribute("id")

        assert isinstance(model_id, ConditionalExpr)
        assert isinstance(model_id.condition, ComparisonExpr)
        assert model_id.condition.operator == "=="
        assert isinstance(model_id.condition.left, StateRef)
        assert model_id.condition.left.path == "$input.env"
        assert model_id.condition.right == "prod"

    def test_parse_conditional_with_number_values(self) -> None:
        """Test parsing conditional with numeric values."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            params {
                temperature = $input.creative ? 0.9 : 0.1
            }
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        params = model.get_params_block()
        assert params is not None
        temp = params.get_attribute("temperature")

        assert isinstance(temp, ConditionalExpr)
        assert temp.true_value == 0.9
        assert temp.false_value == 0.1

    def test_parse_logical_and_expression(self) -> None:
        """Test parsing logical AND expressions."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            enabled = $input.flag1 && $input.flag2
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        enabled = model.get_attribute("enabled")

        assert isinstance(enabled, AndExpr)
        assert len(enabled.operands) == 2
        assert isinstance(enabled.operands[0], StateRef)
        assert isinstance(enabled.operands[1], StateRef)

    def test_parse_logical_or_expression(self) -> None:
        """Test parsing logical OR expressions."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            enabled = $input.flag1 || $input.flag2
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        enabled = model.get_attribute("enabled")

        assert isinstance(enabled, OrExpr)
        assert len(enabled.operands) == 2

    def test_parse_logical_not_expression(self) -> None:
        """Test parsing logical NOT expressions."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            disabled = !$input.enabled
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        disabled = model.get_attribute("disabled")

        assert isinstance(disabled, NotExpr)
        assert isinstance(disabled.operand, StateRef)

    def test_parse_comparison_operators(self) -> None:
        """Test parsing various comparison operators."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            gt = $input.count > 5
            lt = $input.count < 10
            gte = $input.count >= 5
            lte = $input.count <= 10
            ne = $input.status != "error"
        }
        """
        result = parse_agentform(content)
        model = result.models[0]

        gt = model.get_attribute("gt")
        assert isinstance(gt, ComparisonExpr)
        assert gt.operator == ">"

        lt = model.get_attribute("lt")
        assert isinstance(lt, ComparisonExpr)
        assert lt.operator == "<"

        gte = model.get_attribute("gte")
        assert isinstance(gte, ComparisonExpr)
        assert gte.operator == ">="

        lte = model.get_attribute("lte")
        assert isinstance(lte, ComparisonExpr)
        assert lte.operator == "<="

        ne = model.get_attribute("ne")
        assert isinstance(ne, ComparisonExpr)
        assert ne.operator == "!="

    def test_parse_nested_conditional(self) -> None:
        """Test parsing nested conditional expressions."""
        content = """
        agentform { version = "0.1" project = "test" }
        model "gpt4" {
            provider = provider.llm.openai.default
            id = $input.tier == "premium" ? "gpt-4o" : ($input.tier == "standard" ? "gpt-4o-mini" : "gpt-3.5")
        }
        """
        result = parse_agentform(content)
        model = result.models[0]
        model_id = model.get_attribute("id")

        assert isinstance(model_id, ConditionalExpr)
        # The false_value is another conditional
        assert isinstance(model_id.false_value, ConditionalExpr)

    def test_parse_state_ref_in_condition_step(self) -> None:
        """Test parsing state references in condition step."""
        content = """
        agentform { version = "0.1" project = "test" }
        workflow "test" {
            entry = step.check
            step "check" {
                type = "condition"
                condition = $state.result.status == "success"
                on_true = step.success
                on_false = step.failure
            }
            step "success" { type = "end" }
            step "failure" { type = "end" }
        }
        """
        result = parse_agentform(content)
        workflow = result.workflows[0]
        step = workflow.steps[0]
        condition = step.get_attribute("condition")

        assert isinstance(condition, ComparisonExpr)
        assert isinstance(condition.left, StateRef)
        assert condition.left.path == "$state.result.status"
