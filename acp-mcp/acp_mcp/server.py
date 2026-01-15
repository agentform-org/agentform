"""MCP server process management."""

import asyncio
import os
import signal
from typing import Any

import anyio
from anyio.abc import Process

from acp_mcp.types import MCPMethod


class MCPServerManager:
    """Manages MCP server processes."""

    def __init__(self, name: str, command: list[str], auth_token: str | None = None):
        """Initialize server manager.

        Args:
            name: Server name for identification
            command: Command to start the server
            auth_token: Optional authentication token (resolved from env)
        """
        self.name = name
        self.command = command
        self.auth_token = auth_token
        self._process: Process | None = None
        self._tools: list[MCPMethod] = []
        self._request_id = 0

    @property
    def is_running(self) -> bool:
        """Check if server process is running."""
        return self._process is not None and self._process.returncode is None

    def _get_env(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.auth_token:
            # Common env var names for auth tokens
            env["GITHUB_TOKEN"] = self.auth_token
            env["API_TOKEN"] = self.auth_token
        return env

    async def start(self) -> None:
        """Start the MCP server process."""
        if self.is_running:
            return

        self._process = await anyio.open_process(
            self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._get_env(),
        )

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self._process is None:
            return

        try:
            self._process.terminate()
            with anyio.move_on_after(5):
                await self._process.wait()
        except ProcessLookupError:
            pass
        finally:
            self._process = None

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request to the server.

        Args:
            method: The RPC method name
            params: Optional parameters

        Returns:
            The result from the server

        Raises:
            RuntimeError: If server is not running
            Exception: If server returns an error
        """
        import json

        if not self.is_running or self._process is None:
            raise RuntimeError(f"Server {self.name} is not running")

        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(f"Server {self.name} has no stdin/stdout")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # Send request
        request_line = json.dumps(request) + "\n"
        await self._process.stdin.send(request_line.encode())

        # Read response
        response_line = await self._process.stdout.receive()
        response = json.loads(response_line.decode())

        if "error" in response and response["error"]:
            error = response["error"]
            raise Exception(f"MCP error: {error.get('message', 'Unknown error')}")

        return response.get("result")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP connection."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "acp", "version": "0.1.0"},
        }
        return await self.send_request("initialize", params)

    async def list_tools(self) -> list[MCPMethod]:
        """List available tools from the server."""
        result = await self.send_request("tools/list")
        tools = result.get("tools", [])
        self._tools = [MCPMethod(**tool) for tool in tools]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool on the server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        params = {"name": name, "arguments": arguments or {}}
        result = await self.send_request("tools/call", params)
        return result

    @property
    def tools(self) -> list[MCPMethod]:
        """Get cached list of tools."""
        return self._tools

    async def __aenter__(self) -> "MCPServerManager":
        """Context manager entry."""
        await self.start()
        await self.initialize()
        await self.list_tools()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        await self.stop()

