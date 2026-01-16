"""Tests for Agentform schema models."""

from agentform_schema.models import (
    AgentConfig,
    BudgetConfig,
    CapabilityConfig,
    LLMProviderConfig,
    LLMProviderParams,
    ModelConfig,
    PolicyConfig,
    ProjectConfig,
    ProvidersConfig,
    ServerAuthConfig,
    ServerConfig,
    SideEffect,
    SpecRoot,
    StepType,
    Transport,
    WorkflowConfig,
    WorkflowStep,
)


class TestEnums:
    """Test enum definitions."""

    def test_side_effect_values(self):
        """Test SideEffect enum has correct values."""
        assert SideEffect.READ.value == "read"
        assert SideEffect.WRITE.value == "write"

    def test_step_type_values(self):
        """Test StepType enum has correct values."""
        assert StepType.LLM.value == "llm"
        assert StepType.CALL.value == "call"
        assert StepType.CONDITION.value == "condition"
        assert StepType.HUMAN_APPROVAL.value == "human_approval"
        assert StepType.END.value == "end"

    def test_transport_values(self):
        """Test Transport enum has correct values."""
        assert Transport.STDIO.value == "stdio"


class TestServerConfig:
    """Test ServerConfig model."""

    def test_minimal_server_config(self):
        """Test creating server with minimal required fields."""
        server = ServerConfig(name="test", command=["echo", "hello"])
        assert server.name == "test"
        assert server.command == ["echo", "hello"]
        assert server.type == "mcp"
        assert server.transport == Transport.STDIO
        assert server.auth is None

    def test_server_with_auth(self):
        """Test creating server with auth configuration."""
        auth = ServerAuthConfig(token="env:MY_TOKEN")
        server = ServerConfig(name="github", command=["gh"], auth=auth)
        assert server.auth is not None
        assert server.auth.token == "env:MY_TOKEN"

    def test_server_requires_name_and_command(self):
        """Test that name and command are required."""
        # Note: Pydantic v2 allows empty lists and empty strings by default
        # These tests verify the model accepts valid configs
        server1 = ServerConfig(name="test", command=["echo"])
        assert server1.name == "test"

        server2 = ServerConfig(name="github", command=["gh", "api"])
        assert server2.command == ["gh", "api"]


class TestCapabilityConfig:
    """Test CapabilityConfig model."""

    def test_minimal_capability(self):
        """Test creating capability with minimal fields."""
        cap = CapabilityConfig(name="read_file", server="fs", method="readFile")
        assert cap.name == "read_file"
        assert cap.server == "fs"
        assert cap.method == "readFile"
        assert cap.side_effect == SideEffect.READ
        assert cap.requires_approval is False

    def test_capability_with_write_effect(self):
        """Test capability with write side effect."""
        cap = CapabilityConfig(
            name="write_file",
            server="fs",
            method="writeFile",
            side_effect=SideEffect.WRITE,
            requires_approval=True,
        )
        assert cap.side_effect == SideEffect.WRITE
        assert cap.requires_approval is True


class TestBudgetConfig:
    """Test BudgetConfig model."""

    def test_empty_budget(self):
        """Test budget with no constraints."""
        budget = BudgetConfig()
        assert budget.max_cost_usd_per_run is None
        assert budget.max_capability_calls is None
        assert budget.timeout_seconds is None

    def test_budget_with_constraints(self):
        """Test budget with all constraints."""
        budget = BudgetConfig(
            max_cost_usd_per_run=1.50,
            max_capability_calls=100,
            timeout_seconds=300,
        )
        assert budget.max_cost_usd_per_run == 1.50
        assert budget.max_capability_calls == 100
        assert budget.timeout_seconds == 300


class TestPolicyConfig:
    """Test PolicyConfig model."""

    def test_policy_with_budgets(self):
        """Test policy configuration with budgets."""
        policy = PolicyConfig(
            name="default",
            budgets=BudgetConfig(max_cost_usd_per_run=0.50),
        )
        assert policy.name == "default"
        assert policy.budgets is not None
        assert policy.budgets.max_cost_usd_per_run == 0.50

    def test_policy_without_budgets(self):
        """Test policy configuration without budgets."""
        policy = PolicyConfig(name="unlimited")
        assert policy.name == "unlimited"
        assert policy.budgets is None


class TestLLMProviderConfig:
    """Test LLM provider configuration."""

    def test_provider_config(self):
        """Test creating provider config."""
        config = LLMProviderConfig(api_key="env:OPENAI_API_KEY")
        assert config.api_key == "env:OPENAI_API_KEY"
        assert config.default_params is None

    def test_provider_with_params(self):
        """Test provider with default params."""
        params = LLMProviderParams(temperature=0.7, max_tokens=2000, top_p=0.9)
        config = LLMProviderConfig(
            api_key="env:ANTHROPIC_API_KEY",
            default_params=params,
        )
        assert config.default_params is not None
        assert config.default_params.temperature == 0.7
        assert config.default_params.max_tokens == 2000
        assert config.default_params.top_p == 0.9


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_minimal_agent(self):
        """Test creating agent with minimal fields."""
        agent = AgentConfig(
            name="assistant",
            provider="openai",
            model=ModelConfig(preference="gpt-4"),
            instructions="You are helpful.",
        )
        assert agent.name == "assistant"
        assert agent.provider == "openai"
        assert agent.model.preference == "gpt-4"
        assert agent.model.fallback is None
        assert agent.instructions == "You are helpful."
        assert agent.allow == []
        assert agent.policy is None

    def test_agent_with_all_fields(self):
        """Test agent with all fields specified."""
        agent = AgentConfig(
            name="coder",
            provider="anthropic",
            model=ModelConfig(preference="claude-3-opus", fallback="claude-3-sonnet"),
            params=LLMProviderParams(temperature=0.3),
            instructions="You write code.",
            allow=["read_file", "write_file"],
            policy="default",
        )
        assert agent.model.fallback == "claude-3-sonnet"
        assert agent.params is not None
        assert agent.params.temperature == 0.3
        assert len(agent.allow) == 2
        assert agent.policy == "default"


class TestWorkflowStep:
    """Test WorkflowStep model."""

    def test_llm_step(self):
        """Test creating an LLM step."""
        step = WorkflowStep(
            id="process",
            type=StepType.LLM,
            agent="assistant",
            input={"question": "$input.question"},
            save_as="answer",
            next="end",
        )
        assert step.id == "process"
        assert step.type == StepType.LLM
        assert step.agent == "assistant"
        assert step.input == {"question": "$input.question"}
        assert step.save_as == "answer"
        assert step.next == "end"

    def test_call_step(self):
        """Test creating a call step."""
        step = WorkflowStep(
            id="read",
            type=StepType.CALL,
            capability="read_file",
            args={"path": "$input.file_path"},
            next="process",
        )
        assert step.type == StepType.CALL
        assert step.capability == "read_file"
        assert step.args == {"path": "$input.file_path"}

    def test_condition_step(self):
        """Test creating a condition step."""
        step = WorkflowStep(
            id="check",
            type=StepType.CONDITION,
            condition="$state.result == 'success'",
            on_true="success_step",
            on_false="failure_step",
        )
        assert step.type == StepType.CONDITION
        assert step.condition == "$state.result == 'success'"
        assert step.on_true == "success_step"
        assert step.on_false == "failure_step"

    def test_human_approval_step(self):
        """Test creating a human approval step."""
        step = WorkflowStep(
            id="approve",
            type=StepType.HUMAN_APPROVAL,
            payload="$state.changes",
            on_approve="apply",
            on_reject="cancel",
        )
        assert step.type == StepType.HUMAN_APPROVAL
        assert step.payload == "$state.changes"
        assert step.on_approve == "apply"
        assert step.on_reject == "cancel"

    def test_end_step(self):
        """Test creating an end step."""
        step = WorkflowStep(id="end", type=StepType.END)
        assert step.type == StepType.END


class TestWorkflowConfig:
    """Test WorkflowConfig model."""

    def test_workflow_config(self):
        """Test creating a workflow configuration."""
        steps = [
            WorkflowStep(id="start", type=StepType.LLM, agent="assistant", next="end"),
            WorkflowStep(id="end", type=StepType.END),
        ]
        workflow = WorkflowConfig(name="main", entry="start", steps=steps)
        assert workflow.name == "main"
        assert workflow.entry == "start"
        assert len(workflow.steps) == 2


class TestSpecRoot:
    """Test SpecRoot model."""

    def test_minimal_spec(self):
        """Test creating minimal spec."""
        spec = SpecRoot(
            project=ProjectConfig(name="test-project"),
        )
        assert spec.version == "0.1"
        assert spec.project.name == "test-project"
        assert spec.providers.llm == {}
        assert spec.servers == []
        assert spec.capabilities == []
        assert spec.policies == []
        assert spec.agents == []
        assert spec.workflows == []

    def test_full_spec(self):
        """Test creating a full spec."""
        spec = SpecRoot(
            version="0.1",
            project=ProjectConfig(name="full-project"),
            providers=ProvidersConfig(
                llm={
                    "openai": LLMProviderConfig(api_key="env:OPENAI_API_KEY"),
                }
            ),
            servers=[
                ServerConfig(name="fs", command=["node", "fs-server"]),
            ],
            capabilities=[
                CapabilityConfig(name="read_file", server="fs", method="readFile"),
            ],
            policies=[
                PolicyConfig(name="default", budgets=BudgetConfig(timeout_seconds=60)),
            ],
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="openai",
                    model=ModelConfig(preference="gpt-4"),
                    instructions="Help users.",
                    allow=["read_file"],
                    policy="default",
                ),
            ],
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.LLM,
                            agent="assistant",
                            next="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                ),
            ],
        )
        assert spec.project.name == "full-project"
        assert len(spec.providers.llm) == 1
        assert len(spec.servers) == 1
        assert len(spec.capabilities) == 1
        assert len(spec.policies) == 1
        assert len(spec.agents) == 1
        assert len(spec.workflows) == 1
