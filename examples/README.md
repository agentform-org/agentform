# ACP Examples

This directory contains example ACP configurations demonstrating various features and use cases. Each example is a complete, runnable project with its own documentation.

## Quick Start

All examples follow the same pattern:

```bash
cd examples/<example-name>
acp validate --var openai_api_key=test  # Validate the configuration
acp run <workflow> --var openai_api_key=$OPENAI_API_KEY --input-file input.yaml
```

---

## Examples Overview

| Example | Difficulty | Description |
|---------|------------|-------------|
| [simple-agent](#simple-agent) | Beginner | Basic LLM agent answering questions |
| [python-api](#python-api) | Beginner | Using ACP programmatically from Python |
| [multi-agent](#multi-agent) | Intermediate | Multiple agents with conditional routing |
| [filesystem-agent](#filesystem-agent) | Intermediate | File operations via MCP server |
| [pr-reviewer](#pr-reviewer) | Advanced | GitHub PR reviews with human approval |

---

## Simple Agent

**Directory:** [`simple-agent/`](simple-agent/)

The simplest possible ACP configuration. A single agent that answers questions using OpenAI.

### What You'll Learn

- Basic ACP file structure with numbered prefixes
- Defining variables, providers, and models
- Creating an agent with instructions
- Building a simple workflow

### Features Used

- Variables with `sensitive = true`
- LLM provider configuration
- Policy budgets (cost and timeout)
- Fallback models
- Single-step LLM workflow

### Prerequisites

- OpenAI API key

### Run It

```bash
cd examples/simple-agent
acp run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "What is ACP?"}'
```

---

## Python API

**Directory:** [`python-api/`](python-api/)

Use ACP workflows programmatically from Python applications using the `acp-api` package.

### What You'll Learn

- Loading ACP specs from Python code
- Running workflows programmatically
- Using async context managers for resource cleanup
- Error handling patterns
- Integrating ACP into web applications (FastAPI example)

### Features Used

- `acp-api` Python SDK
- Async/await patterns
- Context managers for resource cleanup
- Error handling with custom exceptions
- Web framework integration

### Prerequisites

- Python 3.11+
- `acp-api` package: `pip install acp-api`
- OpenAI API key

### Run It

```bash
cd examples/python-api

# Basic usage
python basic_usage.py

# With context manager
python context_manager.py

# Error handling example
python error_handling.py

# FastAPI web server
pip install fastapi uvicorn
python fastapi_example.py
```

### Use Cases

- Web applications (FastAPI, Flask, Django)
- Background job processors (Celery, RQ)
- Scripts and automation tools
- Testing workflows programmatically

---

## Multi-Agent

**Directory:** [`multi-agent/`](multi-agent/)

Multiple specialized agents working together with intelligent task routing.

### What You'll Learn

- Using multiple LLM providers (OpenAI + Anthropic)
- Creating specialized agents for different tasks
- Conditional workflow routing based on classification
- Differentiated policies for fast vs. thorough processing

### Features Used

- Multiple providers (OpenAI and Anthropic)
- Multiple models across providers
- Different policies (`fast` vs `thorough`)
- Classifier agent for task routing
- Conditional steps (`type = "condition"`)
- State passing between steps

### Prerequisites

- OpenAI API key
- Anthropic API key

### Run It

```bash
cd examples/multi-agent

# Simple task → quick_responder (GPT-4o)
acp run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "What is 2+2?"}'

# Complex task → deep_analyst (Claude)
acp run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "Analyze the impact of AI on employment"}'
```

---

## Filesystem Agent

**Directory:** [`filesystem-agent/`](filesystem-agent/)

An agent that interacts with the filesystem via MCP (Model Context Protocol) server.

### What You'll Learn

- Integrating MCP servers for external capabilities
- Defining capabilities from server methods
- Using `call` steps to invoke capabilities
- Human approval for write operations
- Capability call budgets

### Features Used

- MCP server configuration (`server` block)
- Capability definitions with side effects
- `requires_approval = true` for sensitive operations
- `type = "call"` workflow steps
- `max_capability_calls` budget
- Multiple workflows in one project

### Prerequisites

- OpenAI API key
- Node.js and npm (for MCP filesystem server)

### Run It

```bash
cd examples/filesystem-agent

# Read and summarize a file
acp run read_and_summarize \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"file_path": "article.txt", "task": "Summarize this"}'

# List directory contents
acp run list_and_read \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"directory_path": ".", "task": "What files are here?"}'
```

---

## PR Reviewer

**Directory:** [`pr-reviewer/`](pr-reviewer/)

Automated GitHub pull request reviewer with human-in-the-loop approval.

### What You'll Learn

- GitHub MCP integration with authentication
- Multi-step data gathering workflows
- Human approval gates before write operations
- Building context from multiple API calls
- Safe patterns for automated actions

### Features Used

- MCP server with `auth` block for tokens
- Capabilities: read (`get_pr`, `list_files`) and write (`create_review`)
- `requires_approval = true` on write capabilities
- `type = "human_approval"` workflow step
- Sequential capability calls to build context
- Passing state between steps

### Prerequisites

- OpenAI API key
- GitHub Personal Access Token (with `repo` scope)
- Node.js and npm (for MCP GitHub server)

### Run It

```bash
cd examples/pr-reviewer

acp run review_pr \
  --var openai_api_key=$OPENAI_API_KEY \
  --var github_personal_access_token=$GITHUB_TOKEN \
  --input '{"owner": "your-org", "repo": "your-repo", "pr_number": 123}'
```

The workflow will pause for your approval before submitting the review to GitHub.

---

## File Structure Convention

All examples follow the same file naming convention:

```
example/
├── 00-project.acp      # Project metadata (acp block)
├── 01-variables.acp    # Variable definitions
├── 02-providers.acp    # Provider and model definitions
├── 03-servers.acp      # MCP server configuration (if needed)
├── 04-capabilities.acp # Capability definitions (if needed)
├── 05-policies.acp     # Policy definitions
├── 06-agents.acp       # Agent definitions
├── 07-workflows.acp    # Workflow definitions
├── input.yaml          # Sample input
└── README.md           # Detailed documentation
```

Numbered prefixes ensure files are processed in the correct order. References work across files—for example, agents can reference models defined in the providers file.

---

## Next Steps

After exploring these examples:

1. **Create your own agent**: Start with `simple-agent` as a template
2. **Add capabilities**: Use `filesystem-agent` as a guide for MCP integration
3. **Build complex workflows**: Use `multi-agent` patterns for routing

See the [main README](../README.md) for full documentation.
