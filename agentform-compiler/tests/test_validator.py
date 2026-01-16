"""Tests for Agentform specification validator."""

from agentform_compiler.validator import ValidationResult, validate_spec
from agentform_schema.models import (
    AgentConfig,
    CapabilityConfig,
    LLMProviderConfig,
    ModelConfig,
    PolicyConfig,
    ProjectConfig,
    ProvidersConfig,
    ServerConfig,
    SpecRoot,
    StepType,
    WorkflowConfig,
    WorkflowStep,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_empty_result_is_valid(self):
        """Test that empty result is valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_result_with_error_is_invalid(self):
        """Test that result with error is invalid."""
        result = ValidationResult()
        result.add_error("path.to.field", "Something went wrong")
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].path == "path.to.field"
        assert result.errors[0].message == "Something went wrong"

    def test_result_with_only_warnings_is_valid(self):
        """Test that result with only warnings is still valid."""
        result = ValidationResult()
        result.add_warning("some.path", "This could be improved")
        assert result.is_valid is True
        assert len(result.warnings) == 1


class TestValidateSpec:
    """Tests for validate_spec function."""

    def test_minimal_valid_spec(self):
        """Test validating minimal valid spec."""
        spec = SpecRoot(project=ProjectConfig(name="test"))
        result = validate_spec(spec, check_env=False)
        assert result.is_valid is True

    def test_valid_full_spec(self, monkeypatch):
        """Test validating complete valid spec."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={"openai": LLMProviderConfig(api_key="env:OPENAI_API_KEY")}
            ),
            servers=[ServerConfig(name="fs", command=["node", "fs-server"])],
            capabilities=[CapabilityConfig(name="read_file", server="fs", method="readFile")],
            policies=[PolicyConfig(name="default")],
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="openai",
                    model=ModelConfig(preference="gpt-4"),
                    instructions="Help.",
                    allow=["read_file"],
                    policy="default",
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
                            next="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=True)
        assert result.is_valid is True

    def test_direct_api_key_allowed(self):
        """Test that direct API keys (from variable substitution) are allowed."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(
                llm={"openai": LLMProviderConfig(api_key="sk-hardcoded-key")}
            ),
        )

        result = validate_spec(spec, check_env=False)
        # Direct values are allowed - they come from variable substitution
        assert result.is_valid is True

    def test_capability_references_unknown_server(self):
        """Test that capability referencing unknown server is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            capabilities=[
                CapabilityConfig(
                    name="read_file",
                    server="nonexistent",
                    method="readFile",
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)

    def test_agent_references_unknown_provider(self):
        """Test that agent referencing unknown provider is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            agents=[
                AgentConfig(
                    name="assistant",
                    provider="nonexistent_provider",
                    model=ModelConfig(preference="gpt-4"),
                    instructions="Help.",
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent_provider" in e.message for e in result.errors)

    def test_agent_references_unknown_policy(self):
        """Test that agent referencing unknown policy is rejected."""
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
                    policy="unknown_policy",
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("unknown_policy" in e.message for e in result.errors)

    def test_agent_references_unknown_capability(self):
        """Test that agent referencing unknown capability is rejected."""
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
                    allow=["nonexistent_capability"],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent_capability" in e.message for e in result.errors)

    def test_workflow_unknown_entry_step(self):
        """Test that workflow with unknown entry step is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="nonexistent",
                    steps=[WorkflowStep(id="start", type=StepType.END)],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)

    def test_workflow_step_references_unknown_next(self):
        """Test that step referencing unknown next step is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.END,
                            next="nonexistent",
                        )
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)

    def test_workflow_step_next_to_end_is_valid(self):
        """Test that step with next='end' is valid."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.END,
                            next="end",
                        )
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        # next="end" is a special case that's always valid
        assert result.is_valid is True

    def test_llm_step_missing_agent(self):
        """Test that LLM step without agent is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.LLM,
                            # Missing agent
                        )
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("agent" in e.message.lower() for e in result.errors)

    def test_llm_step_unknown_agent(self):
        """Test that LLM step referencing unknown agent is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.LLM,
                            agent="nonexistent",
                        )
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)

    def test_call_step_missing_capability(self):
        """Test that call step without capability is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.CALL,
                            # Missing capability
                        )
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("capability" in e.message.lower() for e in result.errors)

    def test_condition_step_missing_condition(self):
        """Test that condition step without condition is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.CONDITION,
                            # Missing condition
                            on_true="end",
                            on_false="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("condition" in e.message.lower() for e in result.errors)

    def test_condition_step_unknown_branch(self):
        """Test that condition step with unknown branch is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.CONDITION,
                            condition="$state.x",
                            on_true="nonexistent",
                            on_false="end",
                        ),
                        WorkflowStep(id="end", type=StepType.END),
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("nonexistent" in e.message for e in result.errors)

    def test_human_approval_unknown_branches(self):
        """Test that human approval step with unknown branches is rejected."""
        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            workflows=[
                WorkflowConfig(
                    name="main",
                    entry="start",
                    steps=[
                        WorkflowStep(
                            id="start",
                            type=StepType.HUMAN_APPROVAL,
                            on_approve="unknown1",
                            on_reject="unknown2",
                        ),
                    ],
                )
            ],
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is False
        assert any("unknown1" in e.message for e in result.errors)
        assert any("unknown2" in e.message for e in result.errors)

    def test_missing_env_var_warning(self, monkeypatch):
        """Test that missing env var generates warning."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(llm={"openai": LLMProviderConfig(api_key="env:MISSING_KEY")}),
        )

        result = validate_spec(spec, check_env=True)
        assert result.is_valid is True  # Missing env is warning, not error
        assert len(result.warnings) > 0
        assert any("MISSING_KEY" in w.message for w in result.warnings)

    def test_env_check_disabled(self, monkeypatch):
        """Test that env check can be disabled."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        spec = SpecRoot(
            project=ProjectConfig(name="test"),
            providers=ProvidersConfig(llm={"openai": LLMProviderConfig(api_key="env:MISSING_KEY")}),
        )

        result = validate_spec(spec, check_env=False)
        assert result.is_valid is True
        # No warnings about missing env var when check_env=False
        assert not any("MISSING_KEY" in w.message for w in result.warnings)
