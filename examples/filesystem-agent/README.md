# Filesystem Agent Example

An Agentform example demonstrating an agent that can read, write, and analyze files using the filesystem MCP server.

## Overview

This example showcases how to integrate external capabilities via MCP (Model Context Protocol) servers. The agent can interact with the filesystem to read files, write content, and list directories—all while operating within defined policies and budgets.

## File Structure

```
filesystem-agent/
├── 00-project.agentform       # Project metadata (agentform block)
├── 01-variables.agentform     # Variable definitions
├── 02-providers.agentform     # Provider and model definitions
├── 03-servers.agentform       # MCP server configuration
├── 04-capabilities.agentform  # Capability definitions
├── 05-policies.agentform      # Policy definitions
├── 06-agents.agentform        # Agent definitions
├── 07-workflows.agentform     # Workflow definitions
├── input.yaml           # Sample input
└── README.md
```

## Prerequisites

1. OpenAI API key
2. Node.js and npm (for the MCP filesystem server)

## Usage

Run from the example directory:

```bash
cd examples/filesystem-agent

# Read and summarize a file
agentform run read_and_summarize \
  --var openai_api_key=$OPENAI_API_KEY \
  --input-file input.yaml

# List directory contents
agentform run list_and_read \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"directory_path": ".", "task": "Describe what files are present"}'

# Write a file
agentform run write_file \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"file_path": "notes.txt", "instructions": "Write a brief note about AI assistants"}'
```

To validate:

```bash
agentform validate --var openai_api_key=test
```

## Key Concepts

### MCP Server (`03-servers.agentform`)

Define external MCP servers that provide capabilities:

```hcl
server "filesystem" {
  type      = "mcp"
  transport = "stdio"
  command   = ["npx", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
}
```

### Capabilities (`04-capabilities.agentform`)

Map server methods to named capabilities with defined side effects:

```hcl
capability "read_file" {
  server      = server.filesystem
  method      = "read_file"
  side_effect = "read"
}

capability "write_file" {
  server           = server.filesystem
  method           = "write_file"
  side_effect      = "write"
  requires_approval = true  # Human must approve write operations
}
```

### Policy Budgets (`05-policies.agentform`)

```hcl
policy "filesystem_policy" {
  budgets { max_cost_usd_per_run = 0.50 }
  budgets { max_capability_calls = 20 }
  budgets { timeout_seconds = 120 }
}
```

### Capability Calls vs LLM Steps

- **`type = "call"`**: Directly invokes a capability (file operation, API call)
- **`type = "llm"`**: Sends input to an LLM agent for processing

### State Management

Results from each step are stored in workflow state using `output` blocks:

```hcl
output "file_content" { from = result.data }
```

Subsequent steps reference these with `state.variable_name`.

## Input Schemas

### `read_and_summarize`
```json
{ "file_path": "path/to/file.txt", "task": "Summarize this file" }
```

### `list_and_read`
```json
{ "directory_path": ".", "task": "Describe the directory contents" }
```

### `write_file`
```json
{ "file_path": "output.txt", "instructions": "What content to generate" }
```
