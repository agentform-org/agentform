"""AST (Abstract Syntax Tree) models for ACP native schema.

These models represent the parsed structure of .acp files before
normalization to the existing SpecRoot format.
"""

from typing import Any, cast

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


class VarRef(ASTNode):
    """Variable reference: var.variable_name.

    Used to reference declared variables in the specification.
    """

    var_name: str

    def __str__(self) -> str:
        return f"var.{self.var_name}"


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
    | VarRef
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
        return None

    def get_attributes_dict(self) -> dict[str, Value]:
        """Get all attributes as a dictionary."""
        return {attr.name: attr.value for attr in self.attributes}


class VariableBlock(ASTNode):
    """Variable declaration block.

    variable "openai_api_key" {
        type        = string
        description = "OpenAI API key"
        sensitive   = true
    }

    variable "temperature" {
        type    = number
        default = 0.7
    }
    """

    name: str
    var_type: str | None = None  # string, number, bool, list
    default: Any | None = None
    description: str | None = None
    sensitive: bool = False


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
        api_key = var.openai_api_key
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
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
                # Type assertion: attr.value is Any but should be Value
                return cast("Value", attr.value)
        return None


class ModuleBlock(ASTNode):
    """Module reference block.

    module "llm-provider" {
        source  = "github.com/acp-team/llm-providers"
        version = "v1.2.0"

        // Parameters passed to the module
        openai_api_key = var.openai_api_key
        default_model  = "gpt-4"
    }
    """

    name: str  # Module instance name (e.g., "llm-provider")
    attributes: list[Attribute] = Field(default_factory=list)
    blocks: list[NestedBlock] = Field(default_factory=list)

    def get_attribute(self, name: str) -> Value | None:
        """Get attribute value by name."""
        for attr in self.attributes:
            if attr.name == name:
                return cast("Value", attr.value)
        return None

    @property
    def source(self) -> str | None:
        """Get the module source (Git URL or local path)."""
        val = self.get_attribute("source")
        return val if isinstance(val, str) else None

    @property
    def version(self) -> str | None:
        """Get the module version (Git ref - tag, branch, or commit)."""
        val = self.get_attribute("version")
        return val if isinstance(val, str) else None

    def get_parameters(self) -> dict[str, Value]:
        """Get all parameter attributes (excluding source and version)."""
        params: dict[str, Value] = {}
        for attr in self.attributes:
            if attr.name not in ("source", "version"):
                params[attr.name] = cast("Value", attr.value)
        return params


class ACPFile(ASTNode):
    """Root node representing an entire .acp file.

    Contains all top-level blocks parsed from the file.
    """

    acp: ACPBlock | None = None
    variables: list[VariableBlock] = Field(default_factory=list)
    providers: list[ProviderBlock] = Field(default_factory=list)
    servers: list[ServerBlock] = Field(default_factory=list)
    capabilities: list[CapabilityBlock] = Field(default_factory=list)
    policies: list[PolicyBlock] = Field(default_factory=list)
    models: list[ModelBlock] = Field(default_factory=list)
    agents: list[AgentBlock] = Field(default_factory=list)
    workflows: list[WorkflowBlock] = Field(default_factory=list)
    modules: list[ModuleBlock] = Field(default_factory=list)

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

    def get_variable(self, name: str) -> VariableBlock | None:
        """Get variable by name."""
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def get_module(self, name: str) -> ModuleBlock | None:
        """Get module by name."""
        for module in self.modules:
            if module.name == name:
                return module
        return None


# ============================================================================
# Multi-File Merging
# ============================================================================


class MergeError(Exception):
    """Error during merging of multiple ACP files."""

    def __init__(self, message: str, locations: list[SourceLocation] | None = None):
        self.locations = locations or []
        if self.locations:
            loc_strs = [str(loc) for loc in self.locations]
            super().__init__(f"{message} (in {', '.join(loc_strs)})")
        else:
            super().__init__(message)


def _format_location(loc: SourceLocation | None) -> str:
    """Format a location for error messages."""
    if loc and loc.file:
        return f"{loc.file}:{loc.line}"
    elif loc:
        return f"line {loc.line}"
    return "unknown location"


def merge_acp_files(files: list[ACPFile]) -> ACPFile:
    """Merge multiple ACPFile ASTs into a single ACPFile.

    This function combines all blocks from multiple .acp files into one,
    validating that:
    - Exactly one 'acp {}' metadata block exists across all files
    - No duplicate symbols exist (variables, providers, servers, etc.)

    Args:
        files: List of parsed ACPFile objects to merge

    Returns:
        A single merged ACPFile containing all blocks

    Raises:
        MergeError: If validation fails (multiple acp blocks, no acp block, duplicates)
    """
    if not files:
        raise MergeError("No ACP files to merge")

    if len(files) == 1:
        # Single file, just validate it has an acp block
        if files[0].acp is None:
            raise MergeError(
                "No 'acp' metadata block found. One file must contain an 'acp {}' block."
            )
        return files[0]

    # Collect all acp blocks
    acp_blocks: list[tuple[ACPBlock, SourceLocation | None]] = []
    for f in files:
        if f.acp is not None:
            acp_blocks.append((f.acp, f.acp.location))

    # Validate exactly one acp block
    if len(acp_blocks) == 0:
        raise MergeError("No 'acp' metadata block found. One file must contain an 'acp {}' block.")
    elif len(acp_blocks) > 1:
        locations = [loc for _, loc in acp_blocks if loc]
        loc_strs = [_format_location(loc) for loc in locations]
        raise MergeError(
            f"Multiple 'acp' blocks found: {', '.join(loc_strs)}. Only one is allowed."
        )

    # Create merged file with the single acp block
    merged = ACPFile(acp=acp_blocks[0][0])

    # Track seen symbols for duplicate detection
    seen_variables: dict[str, SourceLocation | None] = {}
    seen_providers: dict[str, SourceLocation | None] = {}
    seen_servers: dict[str, SourceLocation | None] = {}
    seen_capabilities: dict[str, SourceLocation | None] = {}
    seen_policies: dict[str, SourceLocation | None] = {}
    seen_models: dict[str, SourceLocation | None] = {}
    seen_agents: dict[str, SourceLocation | None] = {}
    seen_workflows: dict[str, SourceLocation | None] = {}
    seen_modules: dict[str, SourceLocation | None] = {}

    # Merge all files
    for f in files:
        # Merge variables
        for var in f.variables:
            if var.name in seen_variables:
                existing_loc = _format_location(seen_variables[var.name])
                new_loc = _format_location(var.location)
                raise MergeError(
                    f"Duplicate variable '{var.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_variables[var.name] = var.location
            merged.variables.append(var)

        # Merge providers
        for provider in f.providers:
            key = provider.full_name
            if key in seen_providers:
                existing_loc = _format_location(seen_providers[key])
                new_loc = _format_location(provider.location)
                raise MergeError(
                    f"Duplicate provider '{key}' defined in both {existing_loc} and {new_loc}"
                )
            seen_providers[key] = provider.location
            merged.providers.append(provider)

        # Merge servers
        for server in f.servers:
            if server.name in seen_servers:
                existing_loc = _format_location(seen_servers[server.name])
                new_loc = _format_location(server.location)
                raise MergeError(
                    f"Duplicate server '{server.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_servers[server.name] = server.location
            merged.servers.append(server)

        # Merge capabilities
        for cap in f.capabilities:
            if cap.name in seen_capabilities:
                existing_loc = _format_location(seen_capabilities[cap.name])
                new_loc = _format_location(cap.location)
                raise MergeError(
                    f"Duplicate capability '{cap.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_capabilities[cap.name] = cap.location
            merged.capabilities.append(cap)

        # Merge policies
        for policy in f.policies:
            if policy.name in seen_policies:
                existing_loc = _format_location(seen_policies[policy.name])
                new_loc = _format_location(policy.location)
                raise MergeError(
                    f"Duplicate policy '{policy.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_policies[policy.name] = policy.location
            merged.policies.append(policy)

        # Merge models
        for model in f.models:
            if model.name in seen_models:
                existing_loc = _format_location(seen_models[model.name])
                new_loc = _format_location(model.location)
                raise MergeError(
                    f"Duplicate model '{model.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_models[model.name] = model.location
            merged.models.append(model)

        # Merge agents
        for agent in f.agents:
            if agent.name in seen_agents:
                existing_loc = _format_location(seen_agents[agent.name])
                new_loc = _format_location(agent.location)
                raise MergeError(
                    f"Duplicate agent '{agent.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_agents[agent.name] = agent.location
            merged.agents.append(agent)

        # Merge workflows
        for workflow in f.workflows:
            if workflow.name in seen_workflows:
                existing_loc = _format_location(seen_workflows[workflow.name])
                new_loc = _format_location(workflow.location)
                raise MergeError(
                    f"Duplicate workflow '{workflow.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_workflows[workflow.name] = workflow.location
            merged.workflows.append(workflow)

        # Merge modules
        for module in f.modules:
            if module.name in seen_modules:
                existing_loc = _format_location(seen_modules[module.name])
                new_loc = _format_location(module.location)
                raise MergeError(
                    f"Duplicate module '{module.name}' defined in both {existing_loc} and {new_loc}"
                )
            seen_modules[module.name] = module.location
            merged.modules.append(module)

    return merged
