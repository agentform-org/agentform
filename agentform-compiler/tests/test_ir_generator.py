"""Tests for IR generator."""

import pytest

from agentform_compiler.ir_generator import IRGenerationError, generate_ir
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
    WorkflowConfig,
    WorkflowStep,
)


class TestGenerateIR:
    """Tests for generate_ir function."""

    def test_minimal_spec(self):
        """Test generating IR from minimal spec."""
        spec = SpecRoot(project=ProjectConfig(name="test"))
        ir = generate_ir(spec, resolve_credentials=False)

        assert ir.version == "0.1"
        assert ir.project_name == "test"
        assert ir.providers == {}
        assert ir.servers == {}
        assert ir.capabilities == {}
        assert ir.policies == {}
        assert ir.agents == {}
        assert ir.workflows == {}

    def test_provider_resolution(self, monkeypatch):
        """Test provider resolution."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")

        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={
                    "openai": LLMProviderConfig(
                        api_key="env:OPENAI_API_KEY",
                        default_params=LLMProviderParams(
                            temperature=0.7,
                            max_tokens=2000,
                        ),
                    )
                }
            ),
        )

        ir = generate_ir(spec, resolve_credentials=True)

        assert "openai" in ir.providers
        provider = ir.providers["openai"]
        assert provider.name == "openai"
        assert provider.api_key.env_var == "OPENAI_API_KEY"
        assert provider.api_key.value == "sk-test123"
        assert provider.default_params.temperature == 0.7
        assert provider.default_params.max_tokens == 2000

    def test_provider_without_credential_resolution(self):
        """Test provider without resolving credentials."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={"openai": LLMProviderConfig(api_key="env:OPENAI_API_KEY")}
            ),
        )

        ir = generate_ir(spec, resolve_credentials=False)

        provider = ir.providers["openai"]
        assert provider.api_key.env_var == "OPENAI_API_KEY"
        assert provider.api_key.value is None

    def test_server_resolution(self):
        """Test server resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            servers=[
                ServerConfig(
                    name="fs",
                    command=["npx", "@modelcontextprotocol/server-filesystem"],
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        assert "fs" in ir.servers
        server = ir.servers["fs"]
        assert server.name == "fs"
        assert server.command == ["npx", "@modelcontextprotocol/server-filesystem"]
        assert server.auth_token is None

    def test_server_with_auth_resolution(self, monkeypatch):
        """Test server with auth token resolution."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp-test")

        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            servers=[
                ServerConfig(
                    name="github",
                    command=["gh"],
                    auth=ServerAuthConfig(token="env:GITHUB_TOKEN"),
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=True)

        server = ir.servers["github"]
        assert server.auth_token is not None
        assert server.auth_token.env_var == "GITHUB_TOKEN"
        assert server.auth_token.value == "ghp-test"

    def test_capability_resolution(self):
        """Test capability resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            servers=[ServerConfig(name="fs", command=["node", "fs"])],
            capabilities=[
                CapabilityConfig(
                    name="read_file",
                    server="fs",
                    method="readFile",
                    side_effect=SideEffect.READ,
                    requires_approval=False,
                ),
                CapabilityConfig(
                    name="write_file",
                    server="fs",
                    method="writeFile",
                    side_effect=SideEffect.WRITE,
                    requires_approval=True,
                ),
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        assert len(ir.capabilities) == 2

        read_cap = ir.capabilities["read_file"]
        assert read_cap.server_name == "fs"
        assert read_cap.method_name == "readFile"
        assert read_cap.side_effect == SideEffect.READ
        assert read_cap.requires_approval is False
        assert read_cap.method_schema is None  # Populated during MCP discovery

        write_cap = ir.capabilities["write_file"]
        assert write_cap.side_effect == SideEffect.WRITE
        assert write_cap.requires_approval is True

    def test_policy_resolution(self):
        """Test policy resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            policies=[
                PolicyConfig(
                    name="default",
                    budgets=BudgetConfig(
                        max_cost_usd_per_run=1.00,
                        max_capability_calls=50,
                        timeout_seconds=120,
                    ),
                ),
                PolicyConfig(name="unlimited"),
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        assert len(ir.policies) == 2

        default = ir.policies["default"]
        assert default.budgets.max_cost_usd_per_run == 1.00
        assert default.budgets.max_capability_calls == 50
        assert default.budgets.timeout_seconds == 120

        unlimited = ir.policies["unlimited"]
        assert unlimited.budgets.max_cost_usd_per_run is None

    def test_agent_resolution(self):
        """Test agent resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={
                    "openai": LLMProviderConfig(
                        api_key="env:OPENAI_API_KEY",
                        default_params=LLMProviderParams(
                            temperature=0.7,
                            max_tokens=1000,
                        ),
                    )
                }
            ),
            policies=[PolicyConfig(name="default")],
            capabilities=[CapabilityConfig(name="read_file", server="fs", method="read")],
            servers=[ServerConfig(name="fs", command=["node", "fs"])],
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="openai",
                    model=ModelConfig(preference="gpt-4", fallback="gpt-3.5-turbo"),
                    params=LLMProviderParams(temperature=0.5),  # Override
                    instructions="You are helpful.",
                    allow=["read_file"],
                    policy="default",
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        assert "assistant" in ir.agents
        agent = ir.agents["assistant"]
        assert agent.name == "assistant"
        assert agent.provider_name == "openai"
        assert agent.model_preference == "gpt-4"
        assert agent.model_fallback == "gpt-3.5-turbo"
        assert agent.params.temperature == 0.5  # Overridden
        assert agent.params.max_tokens == 1000  # From provider defaults
        assert agent.instructions == "You are helpful."
        assert agent.allowed_capabilities == ["read_file"]
        assert agent.policy_name == "default"

    def test_agent_inherits_provider_params(self):
        """Test agent inherits provider default params."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={
                    "openai": LLMProviderConfig(
                        api_key="env:OPENAI_API_KEY",
                        default_params=LLMProviderParams(
                            temperature=0.9,
                            max_tokens=500,
                        ),
                    )
                }
            ),
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="openai",
                    model=ModelConfig(preference="gpt-4"),
                    # No params override
                    instructions="Help.",
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        agent = ir.agents["assistant"]
        assert agent.params.temperature == 0.9
        assert agent.params.max_tokens == 500

    def test_agent_unknown_provider_raises(self):
        """Test that unknown provider raises error."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="nonexistent",
                    model=ModelConfig(preference="gpt-4"),
                    instructions="Help.",
                )
            ],
        )

        with pytest.raises(IRGenerationError) as exc_info:
            generate_ir(spec, resolve_credentials=False)
        assert "nonexistent" in str(exc_info.value)

    def test_workflow_resolution(self):
        """Test workflow resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={"openai": LLMProviderConfig(api_key="env:OPENAI_API_KEY")}
            ),
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="openai",
                    model=ModelConfig(preference="gpt-4"),
                    instructions="Help.",
                )
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
                            input={"question": "$input.q"},
                            save_as="result",
                            next="check",
                        ),
                        WorkflowStep(
                            id="check",
                            type=StepType.CONDITION,
                            condition="$state.result",
                            on_true="end",
                            on_false="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        assert "main" in ir.workflows
        workflow = ir.workflows["main"]
        assert workflow.name == "main"
        assert workflow.entry_step == "start"
        assert len(workflow.steps) == 3

        # Steps are indexed by ID
        assert "start" in workflow.steps
        assert "check" in workflow.steps
        assert "end" in workflow.steps

        start = workflow.steps["start"]
        assert start.type == StepType.LLM
        assert start.agent_name == "assistant"
        assert start.input_mapping == {"question": "$input.q"}
        assert start.save_as == "result"
        assert start.next_step == "check"

        check = workflow.steps["check"]
        assert check.type == StepType.CONDITION
        assert check.condition_expr == "$state.result"
        assert check.on_true_step == "end"

    def test_call_step_resolution(self):
        """Test call step resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            servers=[ServerConfig(name="fs", command=["node", "fs"])],
            capabilities=[CapabilityConfig(name="read_file", server="fs", method="read")],
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="read",
                    steps=[
                        WorkflowStep(
                            id="read",
                            type=StepType.CALL,
                            capability="read_file",
                            args={"path": "$input.file"},
                            save_as="content",
                            next="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        read_step = ir.workflows["main"].steps["read"]
        assert read_step.type == StepType.CALL
        assert read_step.capability_name == "read_file"
        assert read_step.args_mapping == {"path": "$input.file"}

    def test_human_approval_step_resolution(self):
        """Test human approval step resolution."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="approve",
                    steps=[
                        WorkflowStep(
                            id="approve",
                            type=StepType.HUMAN_APPROVAL,
                            payload="$state.changes",
                            on_approve="apply",
                            on_reject="cancel",
                        ),
                        WorkflowStep(id="apply", type=StepType.END),
                        WorkflowStep(id="cancel", type=StepType.END),
                    ],
                )
            ],
        )

        ir = generate_ir(spec, resolve_credentials=False)

        approve_step = ir.workflows["main"].steps["approve"]
        assert approve_step.type == StepType.HUMAN_APPROVAL
        assert approve_step.payload_expr == "$state.changes"
        assert approve_step.on_approve_step == "apply"
        assert approve_step.on_reject_step == "cancel"

    def test_direct_api_key_allowed(self):
        """Test that direct API keys (from variable substitution) are allowed."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(llm={"openai": LLMProviderConfig(api_key="plain-text-key")}),
        )

        # Direct values are allowed - they come from variable substitution
        ir = generate_ir(spec, resolve_credentials=False)
        assert ir.providers["openai"].api_key.value == "plain-text-key"
        assert ir.providers["openai"].api_key.env_var == "DIRECT_VALUE"
