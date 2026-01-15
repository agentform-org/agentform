# Filesystem Agent Example

An ACP example demonstrating an agent that can read, write, and analyze files using the filesystem MCP server.

## Overview

This example showcases how to integrate external capabilities via MCP (Model Context Protocol) servers. The agent can interact with the filesystem to read files, write content, and list directories—all while operating within defined policies and budgets.

## Prerequisites

1. OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

### Read and Summarize a File

```bash
acp run read_and_summarize --spec spec.acp --input-file input.yaml
```

### List Directory Contents

```bash
acp run list_and_read --spec spec.acp --input '{
  "directory_path": ".",
  "task": "Describe what files are present"
}'
```

### Write a File

```bash
acp run write_file --spec spec.acp --input '{
  "file_path": "notes.txt",
  "instructions": "Write a brief note about AI assistants"
}'
```

## Spec File Structure

### `server` Block
Define external MCP servers that provide capabilities. The filesystem server runs as a subprocess via stdio transport.

```hcl
server "filesystem" {
  type      = "mcp"                  // Server type (MCP protocol)
  transport = "stdio"                // Communication method
  command   = [
    "npx",
    "@modelcontextprotocol/server-filesystem",
    "/path/to/allowed/directory"    // Restrict access to this path
  ]
}
```

### `capability` Blocks
Map server methods to named capabilities with defined side effects. Side effects help the runtime understand what operations do.

```hcl
capability "read_file" {
  server      = server.filesystem    // Reference to server defined above
  method      = "read_file"          // Method name on the MCP server
  side_effect = "read"               // Indicates read-only operation
}

capability "write_file" {
  server      = server.filesystem
  method      = "write_file"
  side_effect = "write"              // Indicates state-changing operation
}

capability "list_directory" {
  server      = server.filesystem
  method      = "list_directory"
  side_effect = "read"
}
```

### `policy` Block
More granular budgets for agents working with external capabilities.

```hcl
policy "filesystem_policy" {
  budgets { max_cost_usd_per_run = 0.50 }      // Cost limit
  budgets { max_capability_calls = 20 }        // Limit external API calls
  budgets { timeout_seconds = 120 }            // Generous timeout for file operations
}
```

### `model` and `agent` Blocks
Agents reference models and capabilities explicitly.

```hcl
model "gpt4o_mini" {
  provider = provider.llm.openai.default
  id       = "gpt-4o-mini"
}

model "gpt4o" {
  provider = provider.llm.openai.default
  id       = "gpt-4o"
}

agent "file_assistant" {
  model           = model.gpt4o_mini
  fallback_models = [model.gpt4o]

  instructions = <<EOF
You are a helpful file assistant...
EOF

  allow  = [capability.read_file, capability.write_file, capability.list_directory]  // Agent can use these capabilities
  policy = policy.filesystem_policy
}
```

### `workflow` Block
Multiple workflows demonstrate different use cases, each with their own entry points and step chains.

#### Read and Summarize Workflow
```hcl
workflow "read_and_summarize" {
  entry = step.read

  step "read" {
    type       = "call"              // Capability call step
    capability = capability.read_file  // Call this capability

    args { path = input.file_path }  // Pass input to capability

    output "file_content" { from = result.data }  // Store result

    next = step.summarize
  }

  step "summarize" {
    type  = "llm"
    agent = agent.file_assistant

    input {
      file_path = input.file_path
      content   = state.file_content  // Reference previous step's result
      task      = input.task
    }

    output "summary" { from = result.text }

    next = step.end
  }

  step "end" { type = "end" }
}
```

## Input Schemas

### `read_and_summarize` Workflow
```json
{
  "file_path": "path/to/file.txt",
  "task": "Summarize this file"
}
```

### `list_and_read` Workflow
```json
{
  "directory_path": ".",
  "task": "Describe the directory contents"
}
```

### `write_file` Workflow
```json
{
  "file_path": "output.txt",
  "instructions": "What content to generate"
}
```

## Key Concepts

### Capability Calls vs LLM Steps
- **`type = "call"`**: Directly invokes a capability (file operation, API call, etc.)
- **`type = "llm"`**: Sends input to an LLM agent for processing

### State Management
Results from each step are stored in the workflow state using `output` blocks. Subsequent steps can reference these values with `state.variable_name`.

### Side Effects
Capabilities declare their side effects (`read`, `write`) to help the runtime understand operation types. Write operations may require additional approval in some policies.

