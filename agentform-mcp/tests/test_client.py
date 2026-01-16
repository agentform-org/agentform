"""Tests for MCP client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentform_mcp.client import MCPClient
from agentform_mcp.server import MCPServerManager
from agentform_mcp.types import MCPMethod


class TestMCPClient:
    """Tests for MCPClient class."""

    def test_init(self):
        """Test client initialization."""
        client = MCPClient()
        assert client._servers == {}
        assert client.servers == {}

    def test_add_server(self):
        """Test adding a server."""
        client = MCPClient()
        server = client.add_server("test", ["echo", "test"])

        assert "test" in client._servers
        assert isinstance(server, MCPServerManager)
        assert server.name == "test"
        assert server.command == ["echo", "test"]

    def test_add_server_with_auth(self):
        """Test adding a server with auth token."""
        client = MCPClient()
        server = client.add_server("github", ["gh"], auth_token="ghp_token")

        assert server.auth_token == "ghp_token"

    def test_get_server_existing(self):
        """Test getting an existing server."""
        client = MCPClient()
        client.add_server("test", ["echo"])

        server = client.get_server("test")
        assert server is not None
        assert server.name == "test"

    def test_get_server_nonexistent(self):
        """Test getting a non-existent server."""
        client = MCPClient()
        server = client.get_server("nonexistent")
        assert server is None

    def test_servers_property(self):
        """Test servers property returns all servers."""
        client = MCPClient()
        client.add_server("server1", ["cmd1"])
        client.add_server("server2", ["cmd2"])

        servers = client.servers
        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    @pytest.mark.asyncio
    async def test_start_all(self):
        """Test starting all servers."""
        client = MCPClient()

        # Add mock servers
        mock_server1 = AsyncMock(spec=MCPServerManager)
        mock_server2 = AsyncMock(spec=MCPServerManager)
        client._servers = {"s1": mock_server1, "s2": mock_server2}

        await client.start_all()

        mock_server1.start.assert_called_once()
        mock_server1.initialize.assert_called_once()
        mock_server1.list_tools.assert_called_once()
        mock_server2.start.assert_called_once()
        mock_server2.initialize.assert_called_once()
        mock_server2.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all(self):
        """Test stopping all servers."""
        client = MCPClient()

        mock_server1 = AsyncMock(spec=MCPServerManager)
        mock_server2 = AsyncMock(spec=MCPServerManager)
        client._servers = {"s1": mock_server1, "s2": mock_server2}

        await client.stop_all()

        mock_server1.stop.assert_called_once()
        mock_server2.stop.assert_called_once()

    def test_get_all_tools(self):
        """Test getting tools from all servers."""
        client = MCPClient()

        mock_server1 = MagicMock(spec=MCPServerManager)
        mock_server1.tools = [MCPMethod(name="tool1")]
        mock_server2 = MagicMock(spec=MCPServerManager)
        mock_server2.tools = [MCPMethod(name="tool2"), MCPMethod(name="tool3")]

        client._servers = {"s1": mock_server1, "s2": mock_server2}

        all_tools = client.get_all_tools()
        assert len(all_tools) == 2
        assert len(all_tools["s1"]) == 1
        assert len(all_tools["s2"]) == 2

    def test_find_tool_existing(self):
        """Test finding an existing tool."""
        client = MCPClient()

        mock_server = MagicMock(spec=MCPServerManager)
        mock_server.tools = [
            MCPMethod(name="readFile"),
            MCPMethod(name="writeFile"),
        ]
        client._servers = {"fs": mock_server}

        tool = client.find_tool("fs", "readFile")
        assert tool is not None
        assert tool.name == "readFile"

    def test_find_tool_nonexistent_server(self):
        """Test finding tool in non-existent server."""
        client = MCPClient()
        tool = client.find_tool("nonexistent", "tool")
        assert tool is None

    def test_find_tool_nonexistent_tool(self):
        """Test finding non-existent tool."""
        client = MCPClient()

        mock_server = MagicMock(spec=MCPServerManager)
        mock_server.tools = [MCPMethod(name="existingTool")]
        client._servers = {"fs": mock_server}

        tool = client.find_tool("fs", "nonexistent")
        assert tool is None

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test calling a tool successfully."""
        client = MCPClient()

        mock_server = AsyncMock(spec=MCPServerManager)
        mock_server.call_tool.return_value = {"result": "success"}
        client._servers = {"fs": mock_server}

        result = await client.call_tool("fs", "readFile", {"path": "/tmp/test"})

        assert result == {"result": "success"}
        mock_server.call_tool.assert_called_once_with("readFile", {"path": "/tmp/test"})

    @pytest.mark.asyncio
    async def test_call_tool_server_not_found(self):
        """Test calling tool on non-existent server raises error."""
        client = MCPClient()

        with pytest.raises(ValueError) as exc_info:
            await client.call_tool("nonexistent", "tool")
        assert "not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as async context manager."""
        client = MCPClient()
        client.start_all = AsyncMock()  # type: ignore[method-assign]
        client.stop_all = AsyncMock()  # type: ignore[method-assign]

        async with client as ctx:
            assert ctx is client

        client.start_all.assert_called_once()
        client.stop_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_stops_on_exception(self):
        """Test that context manager stops servers on exception."""
        client = MCPClient()
        client.start_all = AsyncMock()  # type: ignore[method-assign]
        client.stop_all = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(RuntimeError):
            async with client:
                raise RuntimeError("Test error")

        client.stop_all.assert_called_once()
