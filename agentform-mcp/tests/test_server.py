"""Tests for MCP server manager."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentform_mcp.server import MCPServerManager
from agentform_mcp.types import MCPMethod


class TestMCPServerManager:
    """Tests for MCPServerManager class."""

    def test_init(self):
        """Test server manager initialization."""
        server = MCPServerManager("test", ["echo", "hello"])

        assert server.name == "test"
        assert server.command == ["echo", "hello"]
        assert server.auth_token is None
        assert server._process is None
        assert server._tools == []
        assert server._request_id == 0

    def test_init_with_auth(self):
        """Test server manager with auth token."""
        server = MCPServerManager("github", ["gh"], auth_token="token123")
        assert server.auth_token == "token123"

    def test_is_running_no_process(self):
        """Test is_running when no process exists."""
        server = MCPServerManager("test", ["echo"])
        assert server.is_running is False

    def test_is_running_with_process(self):
        """Test is_running with active process."""
        server = MCPServerManager("test", ["echo"])
        mock_process = MagicMock()
        mock_process.returncode = None  # Process is running
        server._process = mock_process

        assert server.is_running is True

    def test_is_running_process_exited(self):
        """Test is_running when process has exited."""
        server = MCPServerManager("test", ["echo"])
        mock_process = MagicMock()
        mock_process.returncode = 0  # Process exited
        server._process = mock_process

        assert server.is_running is False

    def test_get_env_without_auth(self):
        """Test environment without auth token."""
        server = MCPServerManager("test", ["echo"])
        env = server._get_env()

        # Should have base environment
        assert "PATH" in env  # Basic env var should be present

    def test_get_env_with_auth(self):
        """Test environment with auth token."""
        server = MCPServerManager("github", ["gh"], auth_token="ghp_test")
        env = server._get_env()

        assert env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_test"
        assert env["GITHUB_TOKEN"] == "ghp_test"
        assert env["API_TOKEN"] == "ghp_test"

    def test_tools_property(self):
        """Test tools property returns cached tools."""
        server = MCPServerManager("test", ["echo"])
        server._tools = [MCPMethod(name="tool1"), MCPMethod(name="tool2")]

        assert len(server.tools) == 2
        assert server.tools[0].name == "tool1"

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test start when already running does nothing."""
        server = MCPServerManager("test", ["echo"])
        mock_process = MagicMock()
        mock_process.returncode = None
        server._process = mock_process

        with patch("anyio.open_process") as mock_open:
            await server.start()
            mock_open.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_creates_process(self):
        """Test start creates a new process."""
        server = MCPServerManager("test", ["echo", "test"])

        mock_process = MagicMock()
        mock_process.returncode = None

        with patch("anyio.open_process", return_value=mock_process) as mock_open:
            await server.start()

            mock_open.assert_called_once()
            assert server._process is mock_process

    @pytest.mark.asyncio
    async def test_stop_no_process(self):
        """Test stop when no process exists."""
        server = MCPServerManager("test", ["echo"])
        # Should not raise
        await server.stop()
        assert server._process is None

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self):
        """Test stop terminates the process."""
        server = MCPServerManager("test", ["echo"])

        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()
        server._process = mock_process

        await server.stop()

        mock_process.terminate.assert_called_once()
        assert server._process is None

    @pytest.mark.asyncio
    async def test_send_request_not_running(self):
        """Test send_request when server not running raises error."""
        server = MCPServerManager("test", ["echo"])

        with pytest.raises(RuntimeError) as exc_info:
            await server.send_request("test/method")
        assert "not running" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_send_request_increments_id(self):
        """Test that send_request increments request ID."""
        server = MCPServerManager("test", ["echo"])

        # Setup mock process
        mock_stdin = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.receive.return_value = b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n'

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        server._process = mock_process

        assert server._request_id == 0
        await server.send_request("test/method")
        assert server._request_id == 1
        await server.send_request("test/method2")
        assert server._request_id == 2

    @pytest.mark.asyncio
    async def test_send_request_formats_json_rpc(self):
        """Test that send_request sends properly formatted JSON-RPC."""
        server = MCPServerManager("test", ["echo"])

        mock_stdin = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.receive.return_value = b'{"jsonrpc": "2.0", "id": 1, "result": {}}\n'

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        server._process = mock_process

        await server.send_request("tools/list", {"cursor": "abc"})

        # Verify the sent data
        sent_data = mock_stdin.send.call_args[0][0]
        sent_json = json.loads(sent_data.decode())

        assert sent_json["jsonrpc"] == "2.0"
        assert sent_json["id"] == 1
        assert sent_json["method"] == "tools/list"
        assert sent_json["params"] == {"cursor": "abc"}

    @pytest.mark.asyncio
    async def test_send_request_handles_error_response(self):
        """Test that send_request raises on error response."""
        server = MCPServerManager("test", ["echo"])

        mock_stdin = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stdout.receive.return_value = b'{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Invalid Request"}}\n'

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        server._process = mock_process

        with pytest.raises(Exception) as exc_info:
            await server.send_request("test/method")
        assert "Invalid Request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize method."""
        server = MCPServerManager("test", ["echo"])
        server.send_request = AsyncMock(return_value={"capabilities": {}})  # type: ignore[method-assign]

        await server.initialize()

        server.send_request.assert_called_once()
        call_args = server.send_request.call_args
        assert call_args[0][0] == "initialize"
        assert "protocolVersion" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test list_tools method."""
        server = MCPServerManager("test", ["echo"])
        server.send_request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "tools": [
                    {"name": "readFile", "description": "Read a file"},
                    {"name": "writeFile", "description": "Write a file"},
                ]
            }
        )

        tools = await server.list_tools()

        server.send_request.assert_called_once_with("tools/list")
        assert len(tools) == 2
        assert tools[0].name == "readFile"
        assert tools[1].name == "writeFile"
        assert server._tools == tools

    @pytest.mark.asyncio
    async def test_call_tool_simple_result(self):
        """Test call_tool with simple text result."""
        server = MCPServerManager("test", ["echo"])
        server._tools = [MCPMethod(name="readFile")]
        server.send_request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "content": [{"type": "text", "text": "File content"}],
                "isError": False,
            }
        )

        result = await server.call_tool("readFile", {"path": "/tmp/test"})

        assert result == "File content"
        server.send_request.assert_called_once_with(
            "tools/call", {"name": "readFile", "arguments": {"path": "/tmp/test"}}
        )

    @pytest.mark.asyncio
    async def test_call_tool_error_result(self):
        """Test call_tool with error result."""
        server = MCPServerManager("test", ["echo"])
        server._tools = [MCPMethod(name="readFile")]
        server.send_request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "content": [{"type": "text", "text": "File not found"}],
                "isError": True,
            }
        )

        with pytest.raises(Exception) as exc_info:
            await server.call_tool("readFile", {"path": "/nonexistent"})
        assert "File not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_multiple_content(self):
        """Test call_tool with multiple content items."""
        server = MCPServerManager("test", ["echo"])
        server._tools = [MCPMethod(name="listFiles")]
        server.send_request = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "content": [
                    {"type": "text", "text": "file1.txt"},
                    {"type": "text", "text": "file2.txt"},
                ],
                "isError": False,
            }
        )

        result = await server.call_tool("listFiles")

        # Should return full content array when multiple items
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_error(self):
        """Test call_tool enhances error for unknown tool."""
        server = MCPServerManager("test", ["echo"])
        server._tools = [MCPMethod(name="existingTool")]
        server.send_request = AsyncMock(side_effect=Exception("Unknown tool: missingTool"))  # type: ignore[method-assign]

        with pytest.raises(Exception) as exc_info:
            await server.call_tool("missingTool")
        assert "Unknown tool" in str(exc_info.value)
        assert "existingTool" in str(exc_info.value)  # Should list available tools

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test server as async context manager."""
        server = MCPServerManager("test", ["echo"])
        server.start = AsyncMock()  # type: ignore[method-assign]
        server.initialize = AsyncMock()  # type: ignore[method-assign]
        server.list_tools = AsyncMock(return_value=[])  # type: ignore[method-assign]
        server.stop = AsyncMock()  # type: ignore[method-assign]

        async with server as ctx:
            assert ctx is server

        server.start.assert_called_once()
        server.initialize.assert_called_once()
        server.list_tools.assert_called_once()
        server.stop.assert_called_once()
