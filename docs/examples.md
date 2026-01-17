---
layout: default
title: Examples
---

# Agentform Examples

The [`examples/`](https://github.com/agentform-team/agentform/tree/main/examples) directory contains ready-to-use configurations demonstrating various Agentform features. Each example includes detailed documentation explaining the concepts it covers.

## Examples Overview

| Example | Difficulty | Description |
|---------|------------|-------------|
| [Simple Agent](#simple-agent) | Beginner | Basic LLM agent answering questions |
| [Multi-Agent](#multi-agent) | Intermediate | Multiple agents with conditional routing |
| [Filesystem Agent](#filesystem-agent) | Intermediate | File operations via MCP server |
| [PR Reviewer](#pr-reviewer) | Advanced | GitHub PR reviews with human approval |

---

## Simple Agent

**Directory:** [`simple-agent/`](https://github.com/agentform-team/agentform/tree/main/examples/simple-agent)

The simplest possible Agentform configuration. A single agent that answers questions using OpenAI.

### What You'll Learn

- Basic Agentform file structure with numbered prefixes
- Defining variables, providers, and models
- Creating an agent with instructions
- Building a simple workflow

### Features Used

- Variables with `sensitive = true`
- LLM provider configuration
- Policy budgets (cost and timeout)
- Fallback models
- Single-step LLM workflow

### Run It

```bash
cd examples/simple-agent
agentform run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "What is Agentform™?"}'
```

---

## Multi-Agent

**Directory:** [`multi-agent/`](https://github.com/agentform-team/agentform/tree/main/examples/multi-agent)

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

### Run It

```bash
cd examples/multi-agent

# Simple task → quick_responder (GPT-4o)
agentform run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "What is 2+2?"}'

# Complex task → deep_analyst (Claude)
agentform run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "Analyze the impact of AI on employment"}'
```

---

## Filesystem Agent

**Directory:** [`filesystem-agent/`](https://github.com/agentform-team/agentform/tree/main/examples/filesystem-agent)

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

### Run It

```bash
cd examples/filesystem-agent

# Read and summarize a file
agentform run read_and_summarize \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"file_path": "article.txt", "task": "Summarize this"}'

# List directory contents
agentform run list_and_read \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"directory_path": ".", "task": "What files are here?"}'
```

---

## PR Reviewer

**Directory:** [`pr-reviewer/`](https://github.com/agentform-team/agentform/tree/main/examples/pr-reviewer)

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

### Run It

```bash
cd examples/pr-reviewer

agentform run review_pr \
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
├── 00-project.agentform      # Project metadata (agentform block)
├── 01-variables.agentform    # Variable definitions
├── 02-providers.agentform    # Provider and model definitions
├── 03-servers.agentform      # MCP server configuration (if needed)
├── 04-capabilities.agentform # Capability definitions (if needed)
├── 05-policies.agentform     # Policy definitions
├── 06-agents.agentform       # Agent definitions
├── 07-workflows.agentform    # Workflow definitions
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
