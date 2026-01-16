"""Pydantic models for Agentform YAML specification."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SideEffect(str, Enum):
    """Side effect classification for capabilities."""

    READ = "read"
    WRITE = "write"


class StepType(str, Enum):
    """Workflow step types."""

    LLM = "llm"
    CALL = "call"
    CONDITION = "condition"
    HUMAN_APPROVAL = "human_approval"
    END = "end"


class Transport(str, Enum):
    """MCP server transport types."""

    STDIO = "stdio"


class ServerAuthConfig(BaseModel):
    """Authentication configuration for MCP servers."""

    token: str | None = None


class ServerConfig(BaseModel):
    """MCP server configuration."""

    name: str
    type: str = "mcp"
    transport: Transport = Transport.STDIO
    command: list[str]
    auth: ServerAuthConfig | None = None


class CapabilityConfig(BaseModel):
    """Capability definition mapping to MCP server methods."""

    name: str
    server: str
    method: str
    side_effect: SideEffect = SideEffect.READ
    requires_approval: bool = False


class BudgetConfig(BaseModel):
    """Budget constraints for policies."""

    max_cost_usd_per_run: float | None = None
    max_capability_calls: int | None = None
    timeout_seconds: int | None = None


class PolicyConfig(BaseModel):
    """Policy configuration for agents."""

    name: str
    budgets: BudgetConfig | None = None


class ModelConfig(BaseModel):
    """Model configuration for agents."""

    preference: str
    fallback: str | None = None


class LLMProviderParams(BaseModel):
    """Default parameters for LLM providers."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


class LLMProviderConfig(BaseModel):
    """LLM provider configuration."""

    api_key: str  # Format: env:VAR_NAME
    default_params: LLMProviderParams | None = None


class ProvidersConfig(BaseModel):
    """Providers configuration section."""

    llm: dict[str, LLMProviderConfig] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent configuration."""

    name: str
    provider: str
    model: ModelConfig
    params: LLMProviderParams | None = None
    instructions: str
    allow: list[str] = Field(default_factory=list)
    policy: str | None = None


class WorkflowStep(BaseModel):
    """Workflow step definition."""

    id: str
    type: StepType

    # For LLM steps
    agent: str | None = None
    input: dict[str, Any] | None = None

    # For call steps
    capability: str | None = None
    args: dict[str, Any] | None = None

    # For condition steps
    condition: str | None = None
    on_true: str | None = None
    on_false: str | None = None

    # For human_approval steps
    payload: str | None = None
    on_approve: str | None = None
    on_reject: str | None = None

    # Common fields
    save_as: str | None = None
    next: str | None = None


class WorkflowConfig(BaseModel):
    """Workflow configuration."""

    name: str
    entry: str
    steps: list[WorkflowStep]


class ProjectConfig(BaseModel):
    """Project configuration."""

    name: str


class SpecRoot(BaseModel):
    """Root of the Agentform YAML specification."""

    version: str = "0.1"
    project: ProjectConfig
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    servers: list[ServerConfig] = Field(default_factory=list)
    capabilities: list[CapabilityConfig] = Field(default_factory=list)
    policies: list[PolicyConfig] = Field(default_factory=list)
    agents: list[AgentConfig] = Field(default_factory=list)
    workflows: list[WorkflowConfig] = Field(default_factory=list)
