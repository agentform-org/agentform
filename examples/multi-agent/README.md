# Multi-Agent Example

An Agentform example demonstrating multiple agents working together with conditional logic to route tasks appropriately.

## Overview

This example showcases advanced Agentform features including:
- **Multiple LLM providers**: OpenAI and Anthropic working together
- **Agent specialization**: Different agents for different task complexities
- **Conditional routing**: Dynamic workflow paths based on classification results
- **Policy differentiation**: Fast vs. thorough processing with different budgets

## File Structure

```
multi-agent/
├── 00-project.agentform      # Project metadata (agentform block)
├── 01-variables.agentform    # Variable definitions
├── 02-providers.agentform    # Provider and model definitions
├── 03-policies.agentform     # Policy definitions
├── 04-agents.agentform       # Agent definitions
├── 05-workflows.agentform    # Workflow definitions
├── input.yaml          # Sample input
└── README.md
```

## Prerequisites

1. OpenAI API key
2. Anthropic API key

## Usage

Run from the example directory:

```bash
cd examples/multi-agent

# Run with provided input
agentform run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input-file input.yaml

# Simple task (routed to quick_responder)
agentform run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "What is 2+2?"}'

# Complex task (routed to deep_analyst)
agentform run smart_respond \
  --var openai_api_key=$OPENAI_API_KEY \
  --var anthropic_api_key=$ANTHROPIC_API_KEY \
  --input '{"task": "Analyze the pros and cons of remote work policies"}'
```

To validate:

```bash
agentform validate --var openai_api_key=test --var anthropic_api_key=test
```

## Key Concepts

### Multiple Providers (`02-providers.agentform`)

Configure different LLM providers:

```hcl
provider "llm.openai" "default" {
  api_key = var.openai_api_key
  default_params { temperature = 0.7 }
}

provider "llm.anthropic" "default" {
  api_key = var.anthropic_api_key
  default_params { temperature = 0.5 }
}
```

### Differentiated Policies (`03-policies.agentform`)

```hcl
policy "fast" {
  budgets { max_cost_usd_per_run = 0.10 }
  budgets { timeout_seconds = 30 }
}

policy "thorough" {
  budgets { max_cost_usd_per_run = 1.00 }
  budgets { timeout_seconds = 120 }
}
```

### Specialized Agents (`04-agents.agentform`)

Each agent has a specific role:

- **classifier**: Quick categorization using GPT-4o-mini
- **quick_responder**: Brief answers using GPT-4o
- **deep_analyst**: Thorough analysis using Claude Sonnet

### Conditional Workflow (`05-workflows.agentform`)

```hcl
step "route" {
  type      = "condition"
  condition = $state.classification.response == "simple"
  on_true   = step.quick_response
  on_false  = step.deep_response
}
```

### Multi-Provider Strategy

Using multiple providers allows you to:
- **Optimize costs**: Use cheaper models for simple tasks
- **Leverage strengths**: Different models excel at different tasks
- **Build resilience**: Fallback options when one provider is unavailable

## Input Schema

```json
{ "task": "Your task or question here" }
```
