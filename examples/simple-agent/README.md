# Simple Agent Example

A minimal ACP example demonstrating a basic LLM-powered agent that answers questions.

## Overview

This is the simplest possible ACP configuration—a single agent with no external capabilities. It showcases the core concepts of ACP specification without the complexity of MCP servers or multi-agent workflows.

## Prerequisites

- OpenAI API key set as environment variable:
  ```bash
  export OPENAI_API_KEY="your-api-key"
  ```

## Usage

Run the example with:

```bash
acp run ask --spec spec.acp --input-file input.yaml
```

Or provide inline input:

```bash
acp run ask --spec spec.acp --input '{"question": "What is the meaning of life?"}'
```

## Spec File Structure

### `acp` Block
Specifies the ACP specification version and project name.

```hcl
acp {
  version = "0.1"
  project = "simple-agent-example"
}
```

### `provider` Block
Configures LLM providers. This example uses OpenAI with default parameters that apply to all agents using this provider.

```hcl
provider "llm.openai" "default" {
  api_key = env("OPENAI_API_KEY")  // Reference environment variable
  default_params {
    temperature = 0.7              // Controls randomness (0-1)
    max_tokens  = 2000             // Maximum response length
  }
}
```

### `policy` Block
Define resource constraints and budgets for agent execution. Policies help prevent runaway costs and ensure predictable behavior.

```hcl
policy "default" {
  budgets { max_cost_usd_per_run = 0.50 }  // Maximum cost per workflow run
  budgets { timeout_seconds = 60 }          // Maximum execution time
}
```

### `model` Blocks
Models are first-class entities attached to providers. They define how to talk to specific LLMs.

```hcl
model "gpt4o_mini" {
  provider = provider.llm.openai.default
  id       = "gpt-4o-mini"
  params {
    temperature = 0.5              // Override provider default
  }
}

model "gpt4o" {
  provider = provider.llm.openai.default
  id       = "gpt-4o"
}
```

### `agent` Block
Configure agents with their models, instructions, and capabilities.

```hcl
agent "assistant" {
  model           = model.gpt4o_mini     // Primary model
  fallback_models = [model.gpt4o]        // Fallback if primary unavailable

  instructions = <<EOF
You are a helpful assistant. Answer questions clearly and concisely.
If you don't know something, say so.
EOF

  policy = policy.default              // Apply the "default" policy
}
```

### `workflow` Block
Define the execution flow using steps. Each workflow has an entry point and a series of connected steps.

```hcl
workflow "ask" {
  entry = step.process                 // Starting step ID

  step "process" {
    type  = "llm"                      // Step that calls an LLM agent
    agent = agent.assistant            // Reference to agent defined above

    input { question = input.question }  // Map workflow input to agent input

    output "answer" { from = result.text }  // Store result in state as "answer"

    next = step.end                    // Next step to execute
  }

  step "end" { type = "end" }          // Terminates the workflow
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

The workflow produces output containing the agent's response, accessible via `$state.answer.response` in subsequent steps or returned as the final result.

