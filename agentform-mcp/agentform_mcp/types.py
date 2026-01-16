"""MCP protocol types."""

from typing import Any

from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 request for MCP."""

    jsonrpc: str = "2.0"
    id: int | str
    method: str
    params: dict[str, Any] | None = None


class MCPError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 response for MCP."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: MCPError | None = None


class MCPMethod(BaseModel):
    """MCP method definition."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = Field(default=None)


class MCPToolsListResult(BaseModel):
    """Result from tools/list method."""

    tools: list[MCPMethod] = Field(default_factory=list)


class MCPInitializeParams(BaseModel):
    """Parameters for initialize method."""

    protocolVersion: str = "2024-11-05"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    clientInfo: dict[str, str] = Field(
        default_factory=lambda: {"name": "agentform", "version": "0.1.0"}
    )


class MCPInitializeResult(BaseModel):
    """Result from initialize method."""

    protocolVersion: str
    capabilities: dict[str, Any] = Field(default_factory=dict)
    serverInfo: dict[str, str] = Field(default_factory=dict)


class MCPCallToolParams(BaseModel):
    """Parameters for tools/call method."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """Result from a tool call."""

    content: list[dict[str, Any]] = Field(default_factory=list)
    isError: bool = False
