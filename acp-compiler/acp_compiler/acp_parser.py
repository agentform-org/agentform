"""Parser for ACP native schema (.acp files).

Uses Lark to parse the grammar and transforms the parse tree
into AST models.
"""

from pathlib import Path
from typing import Any, cast

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedToken

from acp_compiler.acp_ast import (
    ACPBlock,
    ACPFile,
    AgentBlock,
    AndExpr,
    Attribute,
    CapabilityBlock,
    ComparisonExpr,
    ConditionalExpr,
    ModelBlock,
    NestedBlock,
    NotExpr,
    OrExpr,
    PolicyBlock,
    ProviderBlock,
    Reference,
    ServerBlock,
    SourceLocation,
    StateRef,
    StepBlock,
    Value,
    VariableBlock,
    VarRef,
    WorkflowBlock,
    merge_acp_files,
)


class ACPParseError(Exception):
    """Error during ACP parsing."""

    def __init__(
        self,
        message: str,
        line: int | None = None,
        column: int | None = None,
        file: str | None = None,
    ):
        self.line = line
        self.column = column
        self.file = file

        location = ""
        if file:
            location = f"{file}:"
        if line is not None:
            location += f"{line}:"
            if column is not None:
                location += f"{column}:"

        if location:
            super().__init__(f"{location} {message}")
        else:
            super().__init__(message)


def _get_location(meta: Any) -> SourceLocation | None:
    """Extract source location from Lark meta object."""
    if meta is None:
        return None
    try:
        return SourceLocation(
            line=meta.line,
            column=meta.column,
            end_line=meta.end_line,
            end_column=meta.end_column,
        )
    except AttributeError:
        return None


def _get_token_location(token: Token) -> SourceLocation | None:
    """Extract source location from a Lark token."""
    if token is None:
        return None
    try:
        line = token.line
        column = token.column
        if line is None or column is None:
            return None
        return SourceLocation(
            line=line,
            column=column,
            end_line=token.end_line,
            end_column=token.end_column,
        )
    except AttributeError:
        return None


def _unquote(s: str) -> str:
    """Remove surrounding quotes from a string."""
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        # Handle escape sequences
        return s[1:-1].encode().decode("unicode_escape")
    return s


@v_args(meta=True)
class ACPTransformer(Transformer):
    """Transform Lark parse tree into ACP AST."""

    def __init__(self, file_path: str | None = None):
        super().__init__()
        self.file_path = file_path

    def _loc(self, meta: Any) -> SourceLocation | None:
        """Get location with file path."""
        loc = _get_location(meta)
        if loc and self.file_path:
            loc.file = self.file_path
        return loc

    # Start rule - collect all blocks into ACPFile
    def start(self, meta: Any, blocks: list) -> ACPFile:
        acp_file = ACPFile(location=self._loc(meta))

        for block in blocks:
            if isinstance(block, ACPBlock):
                acp_file.acp = block
            elif isinstance(block, VariableBlock):
                acp_file.variables.append(block)
            elif isinstance(block, ProviderBlock):
                acp_file.providers.append(block)
            elif isinstance(block, ServerBlock):
                acp_file.servers.append(block)
            elif isinstance(block, CapabilityBlock):
                acp_file.capabilities.append(block)
            elif isinstance(block, PolicyBlock):
                acp_file.policies.append(block)
            elif isinstance(block, ModelBlock):
                acp_file.models.append(block)
            elif isinstance(block, AgentBlock):
                acp_file.agents.append(block)
            elif isinstance(block, WorkflowBlock):
                acp_file.workflows.append(block)

        return acp_file

    # Pass through block rule
    def block(self, meta: Any, children: list) -> Any:
        return children[0]

    # ACP block
    def acp_block(self, meta: Any, children: list) -> ACPBlock:
        body = children[0] if children else []
        block = ACPBlock(location=self._loc(meta))

        for attr in body:
            if isinstance(attr, Attribute):
                if attr.name == "version" and isinstance(attr.value, str):
                    block.version = attr.value
                elif attr.name == "project" and isinstance(attr.value, str):
                    block.project = attr.value

        return block

    def acp_body(self, meta: Any, children: list) -> list[Attribute]:
        return [c for c in children if isinstance(c, Attribute)]

    # Variable block
    def variable_block(self, meta: Any, children: list) -> VariableBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        # Extract attributes from body
        var_type = None
        default = None
        description = None
        sensitive = False

        for attr in body:
            if isinstance(attr, Attribute):
                if attr.name == "type" and isinstance(attr.value, str):
                    var_type = attr.value
                elif attr.name == "default":
                    default = attr.value
                elif attr.name == "description" and isinstance(attr.value, str):
                    description = attr.value
                elif attr.name == "sensitive" and isinstance(attr.value, bool):
                    sensitive = attr.value

        return VariableBlock(
            name=name,
            var_type=var_type,
            default=default,
            description=description,
            sensitive=sensitive,
            location=self._loc(meta),
        )

    def variable_body(self, meta: Any, children: list) -> list[Attribute]:
        return [c for c in children if isinstance(c, Attribute)]

    # Provider block
    def provider_block(self, meta: Any, children: list) -> ProviderBlock:
        provider_type = _unquote(str(children[0]))
        name = _unquote(str(children[1]))
        body = children[2] if len(children) > 2 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return ProviderBlock(
            provider_type=provider_type,
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def provider_body(self, meta: Any, children: list) -> list:
        return children

    # Server block
    def server_block(self, meta: Any, children: list) -> ServerBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return ServerBlock(
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def server_body(self, meta: Any, children: list) -> list:
        return children

    # Capability block
    def capability_block(self, meta: Any, children: list) -> CapabilityBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return CapabilityBlock(
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def capability_body(self, meta: Any, children: list) -> list:
        return children

    # Policy block
    def policy_block(self, meta: Any, children: list) -> PolicyBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return PolicyBlock(
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def policy_body(self, meta: Any, children: list) -> list:
        return children

    # Model block
    def model_block(self, meta: Any, children: list) -> ModelBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return ModelBlock(
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def model_body(self, meta: Any, children: list) -> list:
        return children

    # Agent block
    def agent_block(self, meta: Any, children: list) -> AgentBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return AgentBlock(
            name=name,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def agent_body(self, meta: Any, children: list) -> list:
        return children

    # Workflow block
    def workflow_block(self, meta: Any, children: list) -> WorkflowBlock:
        name = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        steps = [c for c in body if isinstance(c, StepBlock)]

        return WorkflowBlock(
            name=name,
            attributes=attributes,
            steps=steps,
            location=self._loc(meta),
        )

    def workflow_body(self, meta: Any, children: list) -> list:
        return children

    # Step block
    def step_block(self, meta: Any, children: list) -> StepBlock:
        step_id = _unquote(str(children[0]))
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return StepBlock(
            step_id=step_id,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def step_body(self, meta: Any, children: list) -> list:
        return children

    # Nested blocks
    def unlabeled_nested_block(self, meta: Any, children: list) -> NestedBlock:
        block_type = str(children[0])
        body = children[1] if len(children) > 1 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return NestedBlock(
            block_type=block_type,
            label=None,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def labeled_nested_block(self, meta: Any, children: list) -> NestedBlock:
        block_type = str(children[0])
        label = _unquote(str(children[1]))
        body = children[2] if len(children) > 2 else []

        attributes = [c for c in body if isinstance(c, Attribute)]
        blocks = [c for c in body if isinstance(c, NestedBlock)]

        return NestedBlock(
            block_type=block_type,
            label=label,
            attributes=attributes,
            blocks=blocks,
            location=self._loc(meta),
        )

    def nested_body(self, meta: Any, children: list) -> list:
        return children

    # Attribute
    def attribute(self, meta: Any, children: list) -> Attribute:
        name = str(children[0])
        value = children[1]
        return Attribute(name=name, value=value, location=self._loc(meta))

    # Values
    def heredoc_value(self, meta: Any, children: list) -> str:
        """Parse heredoc string: <<EOF\n...\nEOF"""
        raw = str(children[0])
        # Remove <<EOF\n prefix and \nEOF suffix
        if raw.startswith("<<EOF\n") and raw.endswith("\nEOF"):
            return raw[6:-4]
        elif raw.startswith("<<EOF") and raw.endswith("EOF"):
            return raw[5:-3]
        return raw

    def string_value(self, meta: Any, children: list) -> str:
        return _unquote(str(children[0]))

    def number_value(self, meta: Any, children: list) -> int | float:
        num_str = str(children[0])
        if "." in num_str:
            return float(num_str)
        return int(num_str)

    def boolean_value(self, meta: Any, children: list) -> bool:
        return str(children[0]).lower() == "true"

    def identifier_value(self, meta: Any, children: list) -> str:
        """Parse a bare identifier as a string value (e.g., type = string)."""
        return str(children[0])

    def reference_value(self, meta: Any, children: list) -> Reference:
        result = children[0]
        assert isinstance(result, Reference)
        return result

    def array_value(self, meta: Any, children: list) -> list[Any]:
        result = children[0]
        assert isinstance(result, list)
        return result

    def var_ref_value(self, meta: Any, children: list) -> VarRef:
        result = children[0]
        assert isinstance(result, VarRef)
        return result

    def paren_expr(self, meta: Any, children: list) -> Any:
        """Handle parenthesized expression: (expr)"""
        return children[0]

    def state_ref_value(self, meta: Any, children: list) -> StateRef:
        """Handle state reference value."""
        result = children[0]
        assert isinstance(result, StateRef)
        return result

    # State reference: $input.field or $state.step.field
    def state_ref(self, meta: Any, children: list) -> StateRef:
        """Parse state reference token."""
        path = str(children[0])
        return StateRef(path=path, location=self._loc(meta))

    # Expression rules
    def expr(self, meta: Any, children: list) -> Any:
        """Top-level expression - pass through."""
        return children[0]

    def conditional_expr(self, meta: Any, children: list) -> ConditionalExpr:
        """Conditional expression: condition ? true_val : false_val"""
        condition = children[0]
        true_value = children[1]
        false_value = children[2]
        return ConditionalExpr(
            condition=condition,
            true_value=true_value,
            false_value=false_value,
            location=self._loc(meta),
        )

    def or_expr(self, meta: Any, children: list) -> Any:
        """Logical OR expression."""
        if len(children) == 1:
            return children[0]
        return OrExpr(operands=list(children), location=self._loc(meta))

    def and_expr(self, meta: Any, children: list) -> Any:
        """Logical AND expression."""
        if len(children) == 1:
            return children[0]
        return AndExpr(operands=list(children), location=self._loc(meta))

    def logical_not(self, meta: Any, children: list) -> NotExpr:
        """Logical NOT expression."""
        return NotExpr(operand=children[0], location=self._loc(meta))

    def comparison_expr(self, meta: Any, children: list) -> Any:
        """Comparison expression: left op right."""
        if len(children) == 1:
            return children[0]
        left = children[0]
        operator = str(children[1])
        right = children[2]
        return ComparisonExpr(
            left=left,
            operator=operator,
            right=right,
            location=self._loc(meta),
        )

    def pass_through(self, meta: Any, children: list) -> Any:
        """Pass through a single child unchanged."""
        return children[0]

    # Reference
    def reference(self, meta: Any, children: list) -> Reference:
        parts = [str(c) for c in children]
        return Reference(parts=parts, location=self._loc(meta))

    # Array
    def array(self, meta: Any, children: list) -> list[Value]:
        return list(children)

    # Variable reference
    def var_ref(self, meta: Any, children: list) -> VarRef:
        var_name = str(children[0])
        return VarRef(var_name=var_name, location=self._loc(meta))

    # Terminals
    def STRING(self, token: Token) -> Token:
        return token

    def IDENTIFIER(self, token: Token) -> Token:
        return token

    def SIGNED_NUMBER(self, token: Token) -> Token:
        return token

    def BOOLEAN(self, token: Token) -> Token:
        return token

    def STATE_REF(self, token: Token) -> Token:
        return token

    def COMP_OP(self, token: Token) -> Token:
        return token


# Load grammar from file
_GRAMMAR_PATH = Path(__file__).parent / "acp_grammar.lark"


def _get_parser() -> Lark:
    """Get or create the Lark parser."""
    grammar = _GRAMMAR_PATH.read_text()
    return Lark(
        grammar,
        start="start",
        parser="lalr",
        propagate_positions=True,
        maybe_placeholders=False,
    )


# Cached parser instance
_parser: Lark | None = None


def get_parser() -> Lark:
    """Get the cached parser instance."""
    global _parser
    if _parser is None:
        _parser = _get_parser()
    return _parser


def parse_acp(content: str, file_path: str | None = None) -> ACPFile:
    """Parse ACP content string into an AST.

    Args:
        content: ACP file content as a string
        file_path: Optional file path for error messages

    Returns:
        Parsed ACPFile AST

    Raises:
        ACPParseError: If parsing fails
    """
    parser = get_parser()
    transformer = ACPTransformer(file_path=file_path)

    try:
        tree = parser.parse(content)
        result = transformer.transform(tree)
        return cast("ACPFile", result)
    except UnexpectedCharacters as e:
        raise ACPParseError(
            f"Unexpected character: {e.char!r}",
            line=e.line,
            column=e.column,
            file=file_path,
        ) from e
    except UnexpectedToken as e:
        expected = ", ".join(sorted(e.expected)) if e.expected else "unknown"
        raise ACPParseError(
            f"Unexpected token: {e.token!r}, expected one of: {expected}",
            line=e.line,
            column=e.column,
            file=file_path,
        ) from e
    except LarkError as e:
        raise ACPParseError(f"Parse error: {e}", file=file_path) from e


def parse_acp_file(path: str | Path) -> ACPFile:
    """Parse an ACP file into an AST.

    Args:
        path: Path to the .acp file

    Returns:
        Parsed ACPFile AST

    Raises:
        ACPParseError: If parsing fails or file not found
    """
    path = Path(path)

    if not path.exists():
        raise ACPParseError(f"File not found: {path}")

    try:
        content = path.read_text()
    except OSError as e:
        raise ACPParseError(f"Failed to read file: {e}") from e

    return parse_acp(content, file_path=str(path))


# ============================================================================
# Multi-File / Directory Parsing
# ============================================================================


def discover_acp_files(directory: str | Path) -> list[Path]:
    """Discover all .acp files in a directory.

    Files are sorted alphabetically for consistent ordering.
    Only searches the top-level directory (non-recursive).

    Args:
        directory: Path to directory to search

    Returns:
        List of paths to .acp files, sorted alphabetically

    Raises:
        ACPParseError: If directory does not exist or is not a directory
    """
    directory = Path(directory)

    if not directory.exists():
        raise ACPParseError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ACPParseError(f"Path is not a directory: {directory}")

    # Find all .acp files (case-insensitive extension matching)
    acp_files = [
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".acp"
    ]

    # Sort alphabetically for consistent ordering
    return sorted(acp_files)


def parse_acp_directory(directory: str | Path) -> ACPFile:
    """Parse all .acp files in a directory and merge them into a single AST.

    This function:
    1. Discovers all .acp files in the directory
    2. Parses each file individually
    3. Merges all ASTs into a single ACPFile

    Files are processed in alphabetical order for consistent results.

    Args:
        directory: Path to directory containing .acp files

    Returns:
        Merged ACPFile AST containing all blocks from all files

    Raises:
        ACPParseError: If no .acp files found, parsing fails, or merge fails
        MergeError: If merging fails (duplicate symbols, multiple acp blocks, etc.)
    """
    directory = Path(directory)

    # Discover all .acp files
    acp_files = discover_acp_files(directory)

    if not acp_files:
        raise ACPParseError(f"No .acp files found in directory: {directory}")

    # Parse each file
    parsed_files: list[ACPFile] = []
    for file_path in acp_files:
        parsed_files.append(parse_acp_file(file_path))

    # Merge all files into one
    # MergeError will be raised if validation fails
    return merge_acp_files(parsed_files)
