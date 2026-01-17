---
layout: default
title: Agentform Documentation
---

# Agentform™

<p align="center">
  <strong>Define AI agent systems declaratively using Agentform™ native schema</strong>
</p>

<p align="center">
  Think <em>Infrastructure as Code</em>, but for AI agents
</p>

---

## Why Agentform™?

Most AI agent frameworks require you to write imperative code - managing state, handling retries, wiring up tools. Agentform takes a different approach: **describe your agents declaratively in Agentform native schema, and let the runtime engine handle the rest.**

```hcl
agent "reviewer" {
  model        = model.gpt4o
  instructions = "Review code for security issues"
  allow        = [capability.read_file, capability.get_diff]
  policy       = policy.strict
}
```

**The result:** Your agent configurations become version-controlled artifacts that are easy to review, share, and reproduce. The native `.agentform` format provides type safety, explicit references, and improved editor support.

---

## Quick Start

### 1. Installation

```bash
pip install agentform-cli
```

### 2. Set up your API key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 3. Create an agent spec

Create a file called `my-agent.agentform`:

```hcl
agentform {
  version = "0.1"
  project = "my-first-agent"
}

variable "openai_api_key" {
  type        = string
  description = "OpenAI API key"
  sensitive   = true
}

provider "llm.openai" "default" {
  api_key = var.openai_api_key
  default_params {
    temperature = 0.7
    max_tokens  = 2000
  }
}

policy "default" {
  budgets { max_cost_usd_per_run = 0.50 }
  budgets { timeout_seconds = 60 }
}

model "gpt4o_mini" {
  provider = provider.llm.openai.default
  id       = "gpt-4o-mini"
}

model "gpt4o" {
  provider = provider.llm.openai.default
  id       = "gpt-4o"
}

agent "assistant" {
  model           = model.gpt4o_mini
  fallback_models = [model.gpt4o]

  instructions = "You are a helpful assistant. Answer questions clearly and concisely."

  policy = policy.default
}

workflow "ask" {
  entry = step.process

  step "process" {
    type  = "llm"
    agent = agent.assistant

    input { question = input.question }

    output "answer" { from = result.text }

    next = step.end
  }

  step "end" { type = "end" }
}
```

### 4. Run it

```bash
# Validate your spec
agentform validate my-agent.agentform

# Run with input
agentform run ask --spec my-agent.agentform --input '{"question": "What is the capital of France?"}'
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Native Schema** | Define agents, workflows, and policies in type-safe `.agentform` format with explicit references |
| **Modules** | Terraform-style reusable modules for sharing agent configurations via Git |
| **Multi-Provider** | Use OpenAI, Anthropic, or other LLM providers interchangeably |
| **Multi-Agent** | Coordinate multiple specialized agents with conditional routing |
| **MCP Integration** | Connect to external tools via Model Context Protocol servers |
| **Policy Enforcement** | Set budgets, timeouts, and capability limits per agent |
| **Human-in-the-Loop** | Built-in approval gates for sensitive operations |
| **Execution Tracing** | Full visibility into workflow execution for debugging |

---

## Next Steps

- [Getting Started Guide](/getting-started/) - Detailed installation and setup
- [Examples](/examples/) - Learn from real-world examples
- [Modules](/modules/) - Create and share reusable agent configurations
- [CLI Reference](/cli-reference/) - Complete command reference
- [Architecture](/architecture/) - Understand how Agentform works

---

<p align="center">
  <sub>Built with ❤️ for the AI agent community</sub>
</p>
