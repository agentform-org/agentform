"""Tests for Agentform IR (Intermediate Representation) models."""

from agentform_schema.ir import (
    CompiledSpec,
    MCPMethodSchema,
    ResolvedAgent,
    ResolvedCapability,
    ResolvedCredential,
    ResolvedPolicy,
    ResolvedProvider,
    ResolvedServer,
    ResolvedStep,
    ResolvedWorkflow,
)
from agentform_schema.models import (
    BudgetConfig,
    LLMProviderParams,
    SideEffect,
    StepType,
)


class TestResolvedCredential:
    """Test ResolvedCredential model."""

    def test_credential_without_value(self):
        """Test credential with just env var name."""
        cred = ResolvedCredential(env_var="OPENAI_API_KEY")
        assert cred.env_var == "OPENAI_API_KEY"
        assert cred.value is None

    def test_credential_with_value(self):
        """Test credential with resolved value."""
        cred = ResolvedCredential(env_var="OPENAI_API_KEY", value="sk-test123")
        assert cred.env_var == "OPENAI_API_KEY"
        assert cred.value == "sk-test123"


class TestResolvedProvider:
    """Test ResolvedProvider model."""

    def test_provider(self):
        """Test creating resolved provider."""
        provider = ResolvedProvider(
            name="openai",
            api_key=ResolvedCredential(env_var="OPENAI_API_KEY", value="sk-test"),
            default_params=LLMProviderParams(temperature=0.7),
        )
        assert provider.name == "openai"
        assert provider.api_key.value == "sk-test"
        assert provider.default_params.temperature == 0.7


class TestResolvedServer:
    """Test ResolvedServer model."""

    def test_server_without_auth(self):
        """Test server without auth token."""
        server = ResolvedServer(name="fs", command=["node", "fs-server"])
        assert server.name == "fs"
        assert server.command == ["node", "fs-server"]
        assert server.auth_token is None

    def test_server_with_auth(self):
        """Test server with auth token."""
        auth = ResolvedCredential(env_var="GITHUB_TOKEN", value="ghp_test")
        server = ResolvedServer(name="github", command=["gh"], auth_token=auth)
        assert server.auth_token is not None
        assert server.auth_token.value == "ghp_test"


class TestMCPMethodSchema:
    """Test MCPMethodSchema model."""

    def test_method_schema(self):
        """Test creating method schema."""
        schema = MCPMethodSchema(
            name="readFile",
            description="Read a file from disk",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        assert schema.name == "readFile"
        assert schema.description == "Read a file from disk"
        assert schema.parameters is not None
        assert schema.parameters["required"] == ["path"]

    def test_minimal_method_schema(self):
        """Test method schema with minimal fields."""
        schema = MCPMethodSchema(name="listFiles")
        assert schema.name == "listFiles"
        assert schema.description is None
        assert schema.parameters is None


class TestResolvedCapability:
    """Test ResolvedCapability model."""

    def test_capability(self):
        """Test creating resolved capability."""
        cap = ResolvedCapability(
            name="read_file",
            server_name="fs",
            method_name="readFile",
            method_schema=MCPMethodSchema(name="readFile"),
            side_effect=SideEffect.READ,
            requires_approval=False,
        )
        assert cap.name == "read_file"
        assert cap.server_name == "fs"
        assert cap.method_name == "readFile"
        assert cap.side_effect == SideEffect.READ
        assert cap.requires_approval is False

    def test_capability_without_schema(self):
        """Test capability without method schema."""
        cap = ResolvedCapability(
            name="write_file",
            server_name="fs",
            method_name="writeFile",
            side_effect=SideEffect.WRITE,
            requires_approval=True,
        )
        assert cap.method_schema is None
        assert cap.requires_approval is True


class TestResolvedPolicy:
    """Test ResolvedPolicy model."""

    def test_policy(self):
        """Test creating resolved policy."""
        policy = ResolvedPolicy(
            name="default",
            budgets=BudgetConfig(
                max_cost_usd_per_run=1.00,
                max_capability_calls=50,
                timeout_seconds=120,
            ),
        )
        assert policy.name == "default"
        assert policy.budgets.max_cost_usd_per_run == 1.00
        assert policy.budgets.max_capability_calls == 50
        assert policy.budgets.timeout_seconds == 120


class TestResolvedAgent:
    """Test ResolvedAgent model."""

    def test_agent(self):
        """Test creating resolved agent."""
        agent = ResolvedAgent(
            name="assistant",
            provider_name="openai",
            model_preference="gpt-4",
            model_fallback="gpt-3.5-turbo",
            params=LLMProviderParams(temperature=0.5, max_tokens=1000),
            instructions="You are helpful.",
            allowed_capabilities=["read_file", "write_file"],
            policy_name="default",
        )
        assert agent.name == "assistant"
        assert agent.provider_name == "openai"
        assert agent.model_preference == "gpt-4"
        assert agent.model_fallback == "gpt-3.5-turbo"
        assert agent.params.temperature == 0.5
        assert len(agent.allowed_capabilities) == 2
        assert agent.policy_name == "default"

    def test_agent_minimal(self):
        """Test agent with minimal fields."""
        agent = ResolvedAgent(
            name="simple",
            provider_name="openai",
            model_preference="gpt-4",
            model_fallback=None,
            params=LLMProviderParams(),
            instructions="Basic instructions.",
            allowed_capabilities=[],
            policy_name=None,
        )
        assert agent.model_fallback is None
        assert agent.allowed_capabilities == []
        assert agent.policy_name is None


class TestResolvedStep:
    """Test ResolvedStep model."""

    def test_llm_step(self):
        """Test creating LLM step."""
        step = ResolvedStep(
            id="process",
            type=StepType.LLM,
            agent_name="assistant",
            input_mapping={"question": "$input.q"},
            save_as="answer",
            next_step="end",
        )
        assert step.id == "process"
        assert step.type == StepType.LLM
        assert step.agent_name == "assistant"
        assert step.input_mapping == {"question": "$input.q"}

    def test_call_step(self):
        """Test creating call step."""
        step = ResolvedStep(
            id="read",
            type=StepType.CALL,
            capability_name="read_file",
            args_mapping={"path": "$input.file"},
            next_step="process",
        )
        assert step.type == StepType.CALL
        assert step.capability_name == "read_file"
        assert step.args_mapping == {"path": "$input.file"}

    def test_condition_step(self):
        """Test creating condition step."""
        step = ResolvedStep(
            id="check",
            type=StepType.CONDITION,
            condition_expr="$state.result == 'ok'",
            on_true_step="success",
            on_false_step="failure",
        )
        assert step.type == StepType.CONDITION
        assert step.condition_expr == "$state.result == 'ok'"
        assert step.on_true_step == "success"
        assert step.on_false_step == "failure"

    def test_approval_step(self):
        """Test creating human approval step."""
        step = ResolvedStep(
            id="approve",
            type=StepType.HUMAN_APPROVAL,
            payload_expr="$state.data",
            on_approve_step="apply",
            on_reject_step="cancel",
        )
        assert step.type == StepType.HUMAN_APPROVAL
        assert step.payload_expr == "$state.data"

    def test_end_step(self):
        """Test creating end step."""
        step = ResolvedStep(id="end", type=StepType.END)
        assert step.type == StepType.END


class TestResolvedWorkflow:
    """Test ResolvedWorkflow model."""

    def test_workflow(self):
        """Test creating resolved workflow."""
        steps = {
            "start": ResolvedStep(
                id="start",
                type=StepType.LLM,
                agent_name="assistant",
                next_step="end",
            ),
            "end": ResolvedStep(id="end", type=StepType.END),
        }
        workflow = ResolvedWorkflow(name="main", entry_step="start", steps=steps)
        assert workflow.name == "main"
        assert workflow.entry_step == "start"
        assert len(workflow.steps) == 2
        assert "start" in workflow.steps
        assert "end" in workflow.steps


class TestCompiledSpec:
    """Test CompiledSpec model."""

    def test_empty_compiled_spec(self):
        """Test creating empty compiled spec."""
        spec = CompiledSpec(version="0.1", project_name="test")
        assert spec.version == "0.1"
        assert spec.project_name == "test"
        assert spec.providers == {}
        assert spec.servers == {}
        assert spec.capabilities == {}
        assert spec.policies == {}
        assert spec.agents == {}
        assert spec.workflows == {}

    def test_full_compiled_spec(self):
        """Test creating a full compiled spec."""
        provider = ResolvedProvider(
            name="openai",
            api_key=ResolvedCredential(env_var="OPENAI_API_KEY"),
            default_params=LLMProviderParams(),
        )
        server = ResolvedServer(name="fs", command=["node", "fs-server"])
        capability = ResolvedCapability(
            name="read_file",
            server_name="fs",
            method_name="readFile",
            side_effect=SideEffect.READ,
            requires_approval=False,
        )
        policy = ResolvedPolicy(name="default", budgets=BudgetConfig())
        agent = ResolvedAgent(
            name="assistant",
            provider_name="openai",
            model_preference="gpt-4",
            model_fallback=None,
            params=LLMProviderParams(),
            instructions="Help.",
            allowed_capabilities=["read_file"],
            policy_name="default",
        )
        workflow = ResolvedWorkflow(
            name="main",
            entry_step="start",
            steps={
                "start": ResolvedStep(id="start", type=StepType.END),
            },
        )

        spec = CompiledSpec(
            version="0.1",
            project_name="full-test",
            providers={"openai": provider},
            servers={"fs": server},
            capabilities={"read_file": capability},
            policies={"default": policy},
            agents={"assistant": agent},
            workflows={"main": workflow},
        )

        assert spec.project_name == "full-test"
        assert len(spec.providers) == 1
        assert len(spec.servers) == 1
        assert len(spec.capabilities) == 1
        assert len(spec.policies) == 1
        assert len(spec.agents) == 1
        assert len(spec.workflows) == 1
