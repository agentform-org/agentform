"""AST (Abstract Syntax Tree) models for ACP native schema.

These models represent the parsed structure of .acp files before
normalization to the existing SpecRoot format.
"""

from typing import Any

from pydantic import BaseModel, Field


class SourceLocation(BaseModel):
    """Location in source file for error reporting."""

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None
    file: str | None = None

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        return f"line {self.line}, column {self.column}"


class ASTNode(BaseModel):
    """Base class for all AST nodes."""

    model_config = {"arbitrary_types_allowed": True}

    location: SourceLocation | None = None


class EnvCall(ASTNode):
    """Environment variable reference: env("VAR_NAME")."""

    var_name: str


class Reference(ASTNode):
    """Dotted reference path: provider.llm.openai.default."""

    parts: list[str]

    @property
    def path(self) -> str:
        """Get the full dotted path."""
        return ".".join(self.parts)

    def __str__(self) -> str:
        return self.path


class StateRef(ASTNode):
    """State reference: $input.field or $state.step.field.

    Used in conditional expressions to reference runtime state.
    """

    path: str  # Full path including $, e.g., "$input.name" or "$state.result.value"

    @property
    def parts(self) -> list[str]:
        """Get path parts (excluding the $ prefix)."""
        return self.path[1:].split(".")

    @property
    def root(self) -> str:
        """Get the root (input or state)."""
        return self.parts[0] if self.parts else ""

    def __str__(self) -> str:
        return self.path


class ComparisonExpr(ASTNode):
    """Comparison expression: left op right.

    Examples:
        $state.result == "yes"
        $input.count > 5
    """

    left: Any  # Value type
    operator: str  # ==, !=, <, >, <=, >=
    right: Any  # Value type


class NotExpr(ASTNode):
    """Logical NOT expression: !expr."""

    operand: Any  # Expression to negate


class AndExpr(ASTNode):
    """Logical AND expression: expr && expr."""

    operands: list[Any]  # List of expressions to AND together


class OrExpr(ASTNode):
    """Logical OR expression: expr || expr."""

    operands: list[Any]  # List of expressions to OR together


class ConditionalExpr(ASTNode):
    """Conditional (ternary) expression: condition ? true_val : false_val.

    Examples:
        $input.use_low_temp ? 0.1 : 0.7
        $state.result == "yes" ? step.success : step.failure
    """

    condition: Any  # Expression that evaluates to boolean
    true_value: Any  # Value if condition is true
    false_value: Any  # Value if condition is false


# Value types that can appear in attributes
# Using Any for the recursive list type to avoid Pydantic recursion issues
Value = (
    str
    | int
    | float
    | bool
    | EnvCall
    | Reference
    | StateRef
    | ComparisonExpr
    | NotExpr
    | AndExpr
    | OrExpr
    | ConditionalExpr
    | list[Any]
)


class Attribute(ASTNode):
    """Key-value attribute: key = value."""

    name: str
    value: Any  # Use Any to avoid recursion issues with Value type


class NestedBlock(ASTNode):
    """Nested block with optional label.

    Examples:
        budgets { max_cost = 0.50 }
        output "answer" { from = result.text }
    """

    block_type: str
    label: str | None = None
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list["NestedBlock"] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None

    def get_attributes_dict(self) -> dict[str, Value]:
        """Get all attributes as a dictionary."""
        return {attr.name: attr.value for attr in self.attributes}


class ACPBlock(ASTNode):
    """ACP metadata block.

    acp {
        version = "0.2"
        project = "my-project"
    }
    """

    version: str | None = None
    project: str | None = None


class ProviderBlock(ASTNode):
    """Provider definition block.

    provider "llm.openai" "default" {
        api_key = env("OPENAI_API_KEY")
    }
    """

    provider_type: str  # e.g., "llm.openai"
    name: str  # e.g., "default"
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get the full provider reference name."""
        return f"{self.provider_type}.{self.name}"

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


class ServerBlock(ASTNode):
    """MCP server definition block.

    server "filesystem" {
        type = "mcp"
        transport = "stdio"
        command = ["npx", "@modelcontextprotocol/server-filesystem", "/path"]
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


class CapabilityBlock(ASTNode):
    """Capability definition block.

    capability "read_file" {
        server = server.filesystem
        method = "read_file"
        side_effect = "read"
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


class PolicyBlock(ASTNode):
    """Policy definition block.

    policy "default" {
        budgets { max_cost_usd_per_run = 0.50 }
        budgets { timeout_seconds = 60 }
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_budgets_blocks(self) -> list[NestedBlock]:
        """Get all budget blocks."""
        return [b for b in self.blocks if b.block_type == "budgets"]


class ModelBlock(ASTNode):
    """Model definition block.

    model "openai_gpt4o" {
        provider = provider.llm.openai.default
        id = "gpt-4o"
        params {
            temperature = 0.7
            max_tokens = 2000
        }
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None

    def get_params_block(self) -> NestedBlock | None:
        """Get the params block if present."""
        for block in self.blocks:
            if block.block_type == "params":
                return block
        return None


class AgentBlock(ASTNode):
    """Agent definition block.

    agent "assistant" {
        model = model.openai_gpt4o_mini
        fallback_models = [model.openai_gpt4o]
        instructions = "Answer clearly."
        policy = policy.default
        allow = [capability.read_file]
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


class StepBlock(ASTNode):
    """Workflow step block.

    step "process" {
        type = "llm"
        agent = agent.assistant
        input { question = input.question }
        output "answer" { from = result.text }
        next = step.end
    }
    """

    step_id: str
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None

    def get_input_block(self) -> NestedBlock | None:
        """Get the input block if present."""
        for block in self.blocks:
            if block.block_type == "input":
                return block
        return None

    def get_output_blocks(self) -> list[NestedBlock]:
        """Get all output blocks."""
        return [b for b in self.blocks if b.block_type == "output"]

    def get_args_block(self) -> NestedBlock | None:
        """Get the args block if present (for call steps)."""
        for block in self.blocks:
            if block.block_type == "args":
                return block
        return None


class WorkflowBlock(ASTNode):
    """Workflow definition block.

    workflow "ask" {
        entry = step.process
        step "process" { ... }
        step "end" { type = "end" }
    }
    """

    name: str
    attributes: list[Attribute] = Field(default_factory=list)
    steps: list[StepBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr.value
        return None


class ACPFile(ASTNode):
    """Root node representing an entire .acp file.

    Contains all top-level blocks parsed from the file.
    """

    acp: ACPBlock | None = None
    providers: list[ProviderBlock] = Field(default_factory=list)
    servers: list[ServerBlock] = Field(default_factory=list)
    capabilities: list[CapabilityBlock] = Field(default_factory=list)
    policies: list[PolicyBlock] = Field(default_factory=list)
    models: list[ModelBlock] = Field(default_factory=list)
    agents: list[AgentBlock] = Field(default_factory=list)
    workflows: list[WorkflowBlock] = Field(default_factory=list)

    def get_provider(self, full_name: str) -> ProviderBlock | None:
        """Get provider by full name (e.g., 'llm.openai.default')."""
        for provider in self.providers:
            if provider.full_name == full_name:
                return provider
        return None

    def get_model(self, name: str) -> ModelBlock | None:
        """Get model by name."""
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_agent(self, name: str) -> AgentBlock | None:
        """Get agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_policy(self, name: str) -> PolicyBlock | None:
        """Get policy by name."""
        for policy in self.policies:
            if policy.name == name:
                return policy
        return None

    def get_workflow(self, name: str) -> WorkflowBlock | None:
        """Get workflow by name."""
        for workflow in self.workflows:
            if workflow.name == name:
                return workflow
        return None

    def get_server(self, name: str) -> ServerBlock | None:
        """Get server by name."""
        for server in self.servers:
            if server.name == name:
                return server
        return None

    def get_capability(self, name: str) -> CapabilityBlock | None:
        """Get capability by name."""
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None

