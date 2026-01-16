# Simple Agent Example

A minimal Agentform example demonstrating a basic LLM-powered agent that answers questions.

## Overview

This is the simplest possible Agentform configuration—a single agent with no external capabilities. It showcases the core concepts of Agentform specification without the complexity of MCP servers or multi-agent workflows.

## File Structure

```
simple-agent/
├── 00-project.agentform      # Project metadata (agentform block)
├── 01-variables.agentform    # Variable definitions
├── 02-providers.agentform    # Provider and model definitions
├── 03-policies.agentform     # Policy definitions
├── 04-agents.agentform       # Agent definitions
├── 05-workflows.agentform    # Workflow definitions
├── input.yaml          # Sample input
└── README.md
```

Files are processed in alphabetical order, so we use numbered prefixes to ensure proper ordering. References work across files—for example, `04-agents.agentform` can reference models defined in `02-providers.agentform`.

## Prerequisites

1. OpenAI API key

## Usage

Run from the example directory:

```bash
cd examples/simple-agent

# Run with provided input
agentform run ask --var openai_api_key=$OPENAI_API_KEY --input-file input.yaml

# Or provide inline input
agentform run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "What is the meaning of life?"}'
```

To validate:

```bash
agentform validate --var openai_api_key=test
```

To compile and see the IR:

```bash
agentform compile --var openai_api_key=test
```

## Key Concepts

### Variables (`01-variables.agentform`)

Define variables that can be provided at runtime:

```hcl
variable "openai_api_key" {
  type        = string
  description = "OpenAI API key"
  sensitive   = true
}
```

### Providers & Models (`02-providers.agentform`)

Configure LLM providers and model definitions:

```hcl
provider "llm.openai" "default" {
  api_key = var.openai_api_key
  default_params {
    temperature = 0.7
    max_tokens  = 2000
  }
}

model "gpt4o_mini" {
  provider = provider.llm.openai.default
  id       = "gpt-4o-mini"
  params { temperature = 0.5 }
}
```

### Policies (`03-policies.agentform`)

Define resource constraints and budgets:

```hcl
policy "default" {
  budgets { max_cost_usd_per_run = 0.50 }
  budgets { timeout_seconds = 60 }
}
```

### Agents (`04-agents.agentform`)

Configure agents with models, instructions, and policies:

```hcl
agent "assistant" {
  model           = model.gpt4o_mini
  fallback_models = [model.gpt4o]

  instructions = <<EOF
You are a helpful assistant. Answer questions clearly and concisely.
EOF

  policy = policy.default
}
```

### Workflows (`05-workflows.agentform`)

Define execution flow using steps:

```hcl
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

## Input Schema

```json
{ "question": "Your question here" }
```

## Output

The workflow produces output containing the agent's response, stored in `state.answer`.
