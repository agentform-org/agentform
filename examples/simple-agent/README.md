# Simple Agent Example

A minimal ACP example demonstrating a basic LLM-powered agent that answers questions.

## Overview

This is the simplest possible ACP configuration—a single agent with no external capabilities. It showcases the core concepts of ACP specification without the complexity of MCP servers or multi-agent workflows.

This example also demonstrates **multi-file ACP support** (Terraform-style), where the specification is split across multiple `.acp` files that are automatically merged during compilation.

## File Structure

```
simple-agent/
├── 00-project.acp      # Project metadata (acp block)
├── 01-variables.acp    # Variable definitions
├── 02-providers.acp    # Provider and model definitions
├── 03-policies.acp     # Policy definitions
├── 04-agents.acp       # Agent definitions
├── 05-workflows.acp    # Workflow definitions
├── input.yaml          # Sample input
└── README.md
```

Files are processed in alphabetical order, so we use numbered prefixes to ensure proper ordering. References work across files—for example, `04-agents.acp` can reference models defined in `02-providers.acp`.

## Prerequisites

- OpenAI API key set as environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key"
  ```

## Usage

Run from the example directory (all `.acp` files are discovered automatically):

```bash
cd examples/simple-agent
acp run ask --var openai_api_key=$OPENAI_API_KEY --input-file input.yaml
```

Or provide inline input:

```bash
acp run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "What is the meaning of life?"}'
```

To validate the specification:

```bash
acp validate --var openai_api_key=test
```

To compile and see the IR:

```bash
acp compile --var openai_api_key=test
```

## Spec File Structure

### `00-project.acp` - Project Metadata
Specifies the ACP specification version and project name.

```hcl
acp {
  version = "0.1"
  project = "simple-agent-example"
}
```

### `01-variables.acp` - Variables
Define variables that can be provided at runtime. Sensitive variables (like API keys) should not have defaults.

```hcl
variable "openai_api_key" {
  type        = string
  description = "OpenAI API key"
  sensitive   = true
}
```

### `02-providers.acp` - Providers & Models
Configures LLM providers and model definitions.

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
  params {
    temperature = 0.5
  }
}
```

### `03-policies.acp` - Policies
Define resource constraints and budgets for agent execution.

```hcl
policy "default" {
  budgets { max_cost_usd_per_run = 0.50 }
  budgets { timeout_seconds = 60 }
}
```

### `04-agents.acp` - Agents
Configure agents with their models, instructions, and capabilities.

```hcl
agent "assistant" {
  model           = model.gpt4o_mini
  fallback_models = [model.gpt4o]

  instructions = <<EOF
You are a helpful assistant. Answer questions clearly and concisely.
If you don't know something, say so.
EOF

  policy = policy.default
}
```

### `05-workflows.acp` - Workflows
Define the execution flow using steps.

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

The workflow expects input with the following structure:

```json
{
  "question": "Your question here"
}
```

## Output

The workflow produces output containing the agent's response, accessible via `$state.answer` in subsequent steps or returned as the final result.
