# Agentform MCP

MCP (Model Context Protocol) server integration for Agentform.

## Installation

```bash
poetry install
```

## Usage

```python
from agentform_mcp import MCPClient

async with MCPClient() as client:
    client.add_server("github", ["npx", "@mcp/server-github"])
    await client.start_all()
    result = await client.call_tool("github", "get_repo", {"owner": "...", "repo": "..."})
```

