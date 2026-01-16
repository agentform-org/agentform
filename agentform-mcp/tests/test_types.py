"""Tests for MCP protocol types."""

from agentform_mcp.types import (
    MCPCallToolParams,
    MCPError,
    MCPInitializeParams,
    MCPInitializeResult,
    MCPMethod,
    MCPRequest,
    MCPResponse,
    MCPToolResult,
    MCPToolsListResult,
)


class TestMCPRequest:
    """Tests for MCPRequest model."""

    def test_minimal_request(self):
        """Test creating minimal request."""
        request = MCPRequest(id=1, method="test/method")
        assert request.jsonrpc == "2.0"
        assert request.id == 1
        assert request.method == "test/method"
        assert request.params is None

    def test_request_with_params(self):
        """Test request with parameters."""
        request = MCPRequest(
            id="req-123",
            method="tools/call",
            params={"name": "readFile", "arguments": {"path": "/tmp/test"}},
        )
        assert request.id == "req-123"
        assert request.params is not None
        assert request.params["name"] == "readFile"

    def test_request_id_can_be_string(self):
        """Test that request ID can be a string."""
        request = MCPRequest(id="string-id", method="test")
        assert request.id == "string-id"


class TestMCPError:
    """Tests for MCPError model."""

    def test_basic_error(self):
        """Test creating basic error."""
        error = MCPError(code=-32600, message="Invalid Request")
        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data is None

    def test_error_with_data(self):
        """Test error with additional data."""
        error = MCPError(
            code=-32602,
            message="Invalid params",
            data={"field": "path", "reason": "required"},
        )
        assert error.data is not None
        assert error.data["field"] == "path"


class TestMCPResponse:
    """Tests for MCPResponse model."""

    def test_success_response(self):
        """Test creating success response."""
        response = MCPResponse(id=1, result={"status": "ok"})
        assert response.jsonrpc == "2.0"
        assert response.id == 1
        assert response.result == {"status": "ok"}
        assert response.error is None

    def test_error_response(self):
        """Test creating error response."""
        error = MCPError(code=-32600, message="Error")
        response = MCPResponse(id=1, error=error)
        assert response.error is not None
        assert response.error.code == -32600
        assert response.result is None

    def test_notification_response(self):
        """Test response without ID (notification)."""
        response = MCPResponse(result="notification")
        assert response.id is None


class TestMCPMethod:
    """Tests for MCPMethod model."""

    def test_minimal_method(self):
        """Test creating method with minimal fields."""
        method = MCPMethod(name="readFile")
        assert method.name == "readFile"
        assert method.description is None
        assert method.inputSchema is None

    def test_method_with_all_fields(self):
        """Test method with all fields."""
        method = MCPMethod(
            name="writeFile",
            description="Write content to a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        )
        assert method.name == "writeFile"
        assert method.description == "Write content to a file"
        assert method.inputSchema is not None
        assert method.inputSchema["type"] == "object"
        assert len(method.inputSchema["required"]) == 2


class TestMCPToolsListResult:
    """Tests for MCPToolsListResult model."""

    def test_empty_tools_list(self):
        """Test empty tools list."""
        result = MCPToolsListResult()
        assert result.tools == []

    def test_tools_list_with_methods(self):
        """Test tools list with methods."""
        methods = [
            MCPMethod(name="readFile"),
            MCPMethod(name="writeFile"),
        ]
        result = MCPToolsListResult(tools=methods)
        assert len(result.tools) == 2
        assert result.tools[0].name == "readFile"


class TestMCPInitializeParams:
    """Tests for MCPInitializeParams model."""

    def test_default_params(self):
        """Test default initialization params."""
        params = MCPInitializeParams()
        assert params.protocolVersion == "2024-11-05"
        assert params.capabilities == {}
        assert params.clientInfo["name"] == "agentform"
        assert params.clientInfo["version"] == "0.1.0"

    def test_custom_params(self):
        """Test custom initialization params."""
        params = MCPInitializeParams(
            protocolVersion="2024-12-01",
            capabilities={"tools": {}},
            clientInfo={"name": "custom", "version": "1.0.0"},
        )
        assert params.protocolVersion == "2024-12-01"
        assert "tools" in params.capabilities
        assert params.clientInfo["name"] == "custom"


class TestMCPInitializeResult:
    """Tests for MCPInitializeResult model."""

    def test_result(self):
        """Test initialization result."""
        result = MCPInitializeResult(
            protocolVersion="2024-11-05",
            capabilities={"tools": {"listChanged": True}},
            serverInfo={"name": "test-server", "version": "1.0.0"},
        )
        assert result.protocolVersion == "2024-11-05"
        assert result.capabilities["tools"]["listChanged"] is True
        assert result.serverInfo["name"] == "test-server"

    def test_result_defaults(self):
        """Test result with defaults."""
        result = MCPInitializeResult(protocolVersion="2024-11-05")
        assert result.capabilities == {}
        assert result.serverInfo == {}


class TestMCPCallToolParams:
    """Tests for MCPCallToolParams model."""

    def test_tool_call_without_args(self):
        """Test tool call without arguments."""
        params = MCPCallToolParams(name="listFiles")
        assert params.name == "listFiles"
        assert params.arguments == {}

    def test_tool_call_with_args(self):
        """Test tool call with arguments."""
        params = MCPCallToolParams(
            name="readFile",
            arguments={"path": "/tmp/test.txt", "encoding": "utf-8"},
        )
        assert params.name == "readFile"
        assert params.arguments["path"] == "/tmp/test.txt"


class TestMCPToolResult:
    """Tests for MCPToolResult model."""

    def test_success_result(self):
        """Test successful tool result."""
        result = MCPToolResult(
            content=[{"type": "text", "text": "File content here"}],
            isError=False,
        )
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"

    def test_error_result(self):
        """Test error tool result."""
        result = MCPToolResult(
            content=[{"type": "text", "text": "File not found"}],
            isError=True,
        )
        assert result.isError is True

    def test_empty_result(self):
        """Test empty result."""
        result = MCPToolResult()
        assert result.content == []
        assert result.isError is False

    def test_result_with_multiple_content(self):
        """Test result with multiple content items."""
        result = MCPToolResult(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
                {"type": "image", "data": "base64data"},
            ]
        )
        assert len(result.content) == 3
