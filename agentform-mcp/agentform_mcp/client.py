"""MCP client for managing multiple server connections."""

from typing import Any

from agentform_mcp.server import MCPServerManager
from agentform_mcp.types import MCPMethod


class MCPClient:
    """Client for managing multiple MCP server connections."""

    def __init__(self) -> None:
        """Initialize the MCP client."""
        self._servers: dict[str, MCPServerManager] = {}

    def add_server(
        self,
        name: str,
        command: list[str],
        auth_token: str | None = None,
    ) -> MCPServerManager:
        """Add a server to manage.

        Args:
            name: Server identifier
            command: Command to start the server
            auth_token: Optional auth token

        Returns:
            The server manager instance
        """
        server = MCPServerManager(name, command, auth_token)
        self._servers[name] = server
        return server

    def get_server(self, name: str) -> MCPServerManager | None:
        """Get a server by name."""
        return self._servers.get(name)

    async def start_all(self) -> None:
        """Start all servers and initialize them."""
        for server in self._servers.values():
            await server.start()
            await server.initialize()
            await server.list_tools()

    async def stop_all(self) -> None:
        """Stop all servers."""
        for server in self._servers.values():
            await server.stop()

    def get_all_tools(self) -> dict[str, list[MCPMethod]]:
        """Get all tools from all servers.

        Returns:
            Dict mapping server name to list of tools
        """
        return {name: server.tools for name, server in self._servers.items()}

    def find_tool(self, server_name: str, method_name: str) -> MCPMethod | None:
        """Find a specific tool.

        Args:
            server_name: Server to look in
            method_name: Tool name to find

        Returns:
            The tool if found, None otherwise
        """
        server = self._servers.get(server_name)
        if not server:
            return None

        for tool in server.tools:
            if tool.name == method_name:
                return tool

        return None

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on a specific server.

        Args:
            server_name: Server to call
            tool_name: Tool to invoke
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If server not found
        """
        server = self._servers.get(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")

        return await server.call_tool(tool_name, arguments)

    async def __aenter__(self) -> "MCPClient":
        """Context manager entry."""
        await self.start_all()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.stop_all()

    @property
    def servers(self) -> dict[str, MCPServerManager]:
        """Get all server managers."""
        return self._servers
