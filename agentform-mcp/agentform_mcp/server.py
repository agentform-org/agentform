"""MCP server process management."""

import asyncio
import os
from typing import TYPE_CHECKING, Any, cast

import anyio

from agentform_mcp.types import MCPMethod

if TYPE_CHECKING:
    from anyio.abc import Process


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
            # Set common env var names for auth tokens
            # GitHub MCP server expects GITHUB_PERSONAL_ACCESS_TOKEN
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.auth_token
            # Also set legacy/common names for compatibility
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

        # Read response - MCP uses newline-delimited JSON, so read until we get a complete line
        response_bytes = b""
        while True:
            try:
                chunk = await self._process.stdout.receive()
                if not chunk:
                    break
                response_bytes += chunk
                # Check if we've received a complete line (ends with newline)
                if b"\n" in response_bytes:
                    break
            except EOFError:
                break

        if not response_bytes:
            raise RuntimeError(f"Server {self.name} returned empty response")

        # Extract the first line (up to newline)
        if b"\n" in response_bytes:
            response_bytes = response_bytes.split(b"\n", 1)[0]

        response_str = response_bytes.decode()
        try:
            response = json.loads(response_str)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Server {self.name} returned invalid JSON: {response_str[:100]}"
            ) from e

        if response.get("error"):
            error = response["error"]
            raise Exception(f"MCP error: {error.get('message', 'Unknown error')}")

        return response.get("result")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP connection."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agentform", "version": "0.1.0"},
        }
        result = await self.send_request("initialize", params)
        return cast("dict[str, Any]", result)

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
            Tool result (extracted from content if successful)

        Raises:
            Exception: If tool call resulted in an error
        """
        params = {"name": name, "arguments": arguments or {}}
        try:
            result = await self.send_request("tools/call", params)
        except Exception as e:
            # Enhance error message for "Unknown tool" errors
            if "Unknown tool" in str(e) or "unknown tool" in str(e).lower():
                available_tools = [tool.name for tool in self._tools]
                error_msg = f"{e!s}. Available tools: {', '.join(available_tools) if available_tools else 'none'}"
                raise Exception(error_msg) from e
            raise

        # MCP tool results have structure: {content: [...], isError: bool}
        if not isinstance(result, dict):
            return result

        is_error = result.get("isError", False)
        content = result.get("content", [])

        if is_error:
            # Extract error message from content
            error_messages = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        error_messages.append(item.get("text", ""))
                    elif "error" in item_type.lower():
                        error_messages.append(str(item))

            error_msg = " ".join(error_messages) if error_messages else "Tool call failed"
            raise Exception(f"MCP error: {error_msg}")

        # Extract content - return structured content or extract text
        if content and isinstance(content, list):
            # If there's a single text content item, return just the text
            if (
                len(content) == 1
                and isinstance(content[0], dict)
                and content[0].get("type") == "text"
            ):
                return content[0].get("text")
            # Otherwise return the full content structure
            return content

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
