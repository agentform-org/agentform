"""Intermediate Representation (IR) models for Agentform.

The IR is a compiled, validated, and resolved form of the YAML spec
that the runtime can execute directly.
"""

from typing import Any

from pydantic import BaseModel, Field

from agentform_schema.models import (
    BudgetConfig,
    LLMProviderParams,
    SideEffect,
    StepType,
)


class ResolvedCredential(BaseModel):
    """A resolved credential (from env var)."""

    env_var: str
    value: str | None = None  # Populated at runtime


class ResolvedProvider(BaseModel):
    """A resolved LLM provider configuration."""

    name: str
    api_key: ResolvedCredential
    default_params: LLMProviderParams


class ResolvedServer(BaseModel):
    """A resolved MCP server configuration."""

    name: str
    command: list[str]
    auth_token: ResolvedCredential | None = None


class MCPMethodSchema(BaseModel):
    """Schema for an MCP method discovered from a server."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ResolvedCapability(BaseModel):
    """A resolved capability with discovered method schema."""

    name: str
    server_name: str
    method_name: str
    method_schema: MCPMethodSchema | None = None
    side_effect: SideEffect
    requires_approval: bool


class ResolvedPolicy(BaseModel):
    """A resolved policy configuration."""

    name: str
    budgets: BudgetConfig


class ResolvedAgent(BaseModel):
    """A resolved agent configuration."""

    name: str
    provider_name: str
    model_preference: str
    model_fallback: str | None
    params: LLMProviderParams
    instructions: str
    allowed_capabilities: list[str]
    policy_name: str | None


class ResolvedStep(BaseModel):
    """A resolved workflow step."""

    id: str
    type: StepType

    # For LLM steps
    agent_name: str | None = None
    input_mapping: dict[str, str] | None = None

    # For call steps
    capability_name: str | None = None
    args_mapping: dict[str, Any] | None = None

    # For condition steps
    condition_expr: str | None = None
    on_true_step: str | None = None
    on_false_step: str | None = None

    # For human_approval steps
    payload_expr: str | None = None
    on_approve_step: str | None = None
    on_reject_step: str | None = None

    # Common fields
    save_as: str | None = None
    next_step: str | None = None


class ResolvedWorkflow(BaseModel):
    """A resolved workflow configuration."""

    name: str
    entry_step: str
    steps: dict[str, ResolvedStep]  # Indexed by step ID for fast lookup


class CompiledSpec(BaseModel):
    """The fully compiled and resolved specification (IR)."""

    version: str
    project_name: str
    providers: dict[str, ResolvedProvider] = Field(default_factory=dict)
    servers: dict[str, ResolvedServer] = Field(default_factory=dict)
    capabilities: dict[str, ResolvedCapability] = Field(default_factory=dict)
    policies: dict[str, ResolvedPolicy] = Field(default_factory=dict)
    agents: dict[str, ResolvedAgent] = Field(default_factory=dict)
    workflows: dict[str, ResolvedWorkflow] = Field(default_factory=dict)
