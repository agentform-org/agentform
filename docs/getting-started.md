---
layout: default
title: Getting Started
---

# Getting Started with Agentform

This guide will help you get up and running with Agentform in minutes.

## Installation

### Quick Install (Recommended)

```bash
pip install agentform-cli
```

That's it! You're ready to go.

### Verify Installation

```bash
agentform --help
```

You should see the Agentform CLI help output.

## Prerequisites

- Python 3.12 or higher
- An API key for at least one LLM provider (OpenAI, Anthropic, etc.)

## Your First Agent

Let's create a simple agent that can answer questions.

### Step 1: Set up your API key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### Step 2: Create your first Agentform file

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

### Step 3: Validate your configuration

```bash
agentform validate my-agent.agentform --var openai_api_key=$OPENAI_API_KEY
```

This checks that your configuration is valid and all references are correct.

### Step 4: Run your agent

```bash
agentform run ask \
  --spec my-agent.agentform \
  --var openai_api_key=$OPENAI_API_KEY \
  --input '{"question": "What is the capital of France?"}'
```

You should see the agent's response!

## Understanding the Configuration

Let's break down what we just created:

### `agentform` block
Defines the project metadata and version.

### `variable` block
Declares input variables. The `sensitive = true` flag ensures the value isn't logged.

### `provider` block
Configures the LLM provider (OpenAI in this case) with your API key and default parameters.

### `policy` block
Sets budgets and limits for agent execution:
- `max_cost_usd_per_run`: Maximum cost per workflow run
- `timeout_seconds`: Maximum execution time

### `model` block
Defines specific models from your provider. You can reference multiple models.

### `agent` block
Creates an agent with:
- A primary model (`model.gpt4o_mini`)
- Fallback models if the primary fails
- Instructions that define the agent's behavior
- A policy for resource limits

### `workflow` block
Defines the execution flow:
- `entry`: The starting step
- `step`: Individual workflow steps
  - `type = "llm"`: Uses an LLM agent
  - `input`: Maps input data
  - `output`: Extracts results
  - `next`: The next step in the flow

## Next Steps

- Explore the [Examples](/examples/) to see more complex configurations
- Learn about [Modules](/modules/) for reusable agent configurations
- Check the [CLI Reference](/cli-reference/) for all available commands
