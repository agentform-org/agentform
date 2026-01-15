"""Parser for ACP native schema (.acp files).

Uses Lark to parse the grammar and transforms the parse tree
into AST models.
"""

from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedCharacters, UnexpectedToken

from acp_compiler.acp_ast import (
    ACPBlock,
    ACPFile,
    AgentBlock,
    Attribute,
    CapabilityBlock,
    EnvCall,
    ModelBlock,
    NestedBlock,
    PolicyBlock,
    ProviderBlock,
    Reference,
    ServerBlock,
    SourceLocation,
    StepBlock,
    Value,
    WorkflowBlock,
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
        return SourceLocation(
            line=token.line,
            column=token.column,
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

    def reference_value(self, meta: Any, children: list) -> Reference:
        return children[0]

    def array_value(self, meta: Any, children: list) -> list:
        return children[0]

    def env_value(self, meta: Any, children: list) -> EnvCall:
        return children[0]

    # Reference
    def reference(self, meta: Any, children: list) -> Reference:
        parts = [str(c) for c in children]
        return Reference(parts=parts, location=self._loc(meta))

    # Array
    def array(self, meta: Any, children: list) -> list[Value]:
        return list(children)

    # Env call
    def env_call(self, meta: Any, children: list) -> EnvCall:
        var_name = _unquote(str(children[0]))
        return EnvCall(var_name=var_name, location=self._loc(meta))

    # Terminals
    def STRING(self, token: Token) -> Token:
        return token

    def IDENTIFIER(self, token: Token) -> Token:
        return token

    def SIGNED_NUMBER(self, token: Token) -> Token:
        return token

    def BOOLEAN(self, token: Token) -> Token:
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
        return transformer.transform(tree)
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

