<p align="center">
  <img width="1200" height="300" alt="Agentform" src="https://github.com/user-attachments/assets/40707d9d-ea9c-448b-8c0e-d7b6c1a14140" />
</p>

<p align="center">
  <strong>Define AI agent systems declaratively using Agentform™ native schema</strong>
</p>

<p align="center">
  Think <em>Infrastructure as Code</em>, but for AI agents
</p>

<br />

<p align="center">
  <a href="https://github.com/agentform-org/agentform/actions/workflows/CI.yml?branch=main"><img src="https://img.shields.io/github/actions/workflow/status/agentform-org/agentform/ci.yml?branch=main&style=for-the-badge" alt="CI status"></a>
  <a href="https://github.com/agentform-org/agentform/releases"><img src="https://img.shields.io/github/v/release/agentform-org/agentform?include_prereleases&style=for-the-badge" alt="GitHub release"></a>
  <a href="https://deepwiki.com/agentform-org/agentform"><img src="https://img.shields.io/badge/DeepWiki-agentform-111111?style=for-the-badge" alt="DeepWiki"></a>
  <a href="https://discord.gg/clawd"><img src="https://img.shields.io/discord/1456350064065904867?label=Discord&logo=discord&logoColor=white&color=5865F2&style=for-the-badge" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="Apache 2.0 License"></a>
</p>

git c<p align="center">
  <video width="800" controls>
    <source src="docs/assets/Agentform Promo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

---

## Table of Contents

- [Why Agentform?](#why-agentform)
- [Installation](#installation)
  - [Quick Install (Recommended)](#quick-install-recommended)
  - [Verify Installation](#verify-installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Examples](#examples)
- [Modules](#modules)
  - [Using Modules](#using-modules)
  - [Creating Modules](#creating-modules)
  - [Module Source Formats](#module-source-formats)
  - [Module Caching](#module-caching)
- [CLI Reference](#cli-reference)
- [Contributing](#contributing)
  - [Development Setup](#development-setup)
  - [Project Structure](#project-structure)
- [License](#license)

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

**The result:** Your agent configurations become version-controlled artifacts that are easy to review, share, and reproduce. The native `.af` format provides type safety, explicit references, and improved editor support.

<br />

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

<br />

## Quick Start

### 1. Set up your API key

```bash
export OPENAI_API_KEY="your-openai-key"
```

### 2. Create an agent spec

Create a file called `my-agent.af`:

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

### 3. Run it

```bash
# Validate your spec
agentform validate my-agent.af

# Run with input
agentform run ask --spec my-agent.af --input '{"question": "What is the capital of France?"}'
```

<br />

## Features

| Feature | Description |
|---------|-------------|
| **Native Schema** | Define agents, workflows, and policies in type-safe `.af` format with explicit references |
| **Modules** | Terraform-style reusable modules for sharing agent configurations via Git |
| **Multi-Provider** | Use OpenAI, Anthropic, or other LLM providers interchangeably |
| **Multi-Agent** | Coordinate multiple specialized agents with conditional routing |
| **MCP Integration** | Connect to external tools via Model Context Protocol servers |
| **Policy Enforcement** | Set budgets, timeouts, and capability limits per agent |
| **Human-in-the-Loop** | Built-in approval gates for sensitive operations |
| **Execution Tracing** | Full visibility into workflow execution for debugging |

<br />

## Architecture

Agentform is built as a modular system with five core packages:

```mermaid
flowchart TB
    subgraph User["User Layer"]
        Agentform["📄 .af Spec"]
        CLI["⚡ agentform-cli"]
        Agentform --> CLI
    end

    subgraph Core["Core Layer"]
        SCHEMA["📦 agentform-schema"]
        COMPILER["🔧 agentform-compiler"]
        RUNTIME["🚀 agentform-runtime"]
        COMPILER --> SCHEMA
        COMPILER --> RUNTIME
    end

    subgraph Integration["Integration Layer"]
        MCP["🔌 agentform-mcp"]
        LLM["🧠 LLM Providers"]
        TOOLS["🛠️ External Tools"]
    end

    CLI --> COMPILER
    RUNTIME --> MCP
    RUNTIME --> LLM
    MCP --> TOOLS
```

| Package | Description |
|---------|-------------|
| **agentform-schema** | Core Pydantic models for specs and Intermediate Representation |
| **agentform-compiler** | Parses `.af` files, validates specs, and generates IR for the runtime |
| **agentform-runtime** | Workflow execution engine with LLM integration and policy enforcement |
| **agentform-mcp** | MCP (Model Context Protocol) client for connecting to external tool servers |
| **agentform-cli** | Command-line interface for validating and running workflows |

<br />

## Examples

The [`examples/`](examples/) directory contains ready-to-use configurations demonstrating various Agentform features. Each example includes detailed documentation explaining the concepts it covers.

**[Browse all examples →](examples/)**

Quick start with the simplest example:

```bash
cd examples/simple-agent
agentform run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "Hello!"}'
```

<br />

## Modules

Agentform supports a **Terraform-style module system** for creating reusable, shareable agent configurations. Modules let you package providers, policies, agents, and workflows together, making it easy for others to use without extensive configuration.

### Using Modules

#### 1. Import a module in your project

Create a module block referencing a Git repository:

```hcl
module "pr-reviewer" {
  source  = "github.com/org/agentform-modules//pr-reviewer"
  version = "v1.0.0"  // Git branch, tag, or commit
  
  // Pass required parameters
  api_key = var.openai_api_key
  model   = "gpt-4o"
}
```

The `//` syntax separates the repository URL from the subdirectory path (like Terraform).

#### 2. Initialize your project

Download all external modules to your local `.af/modules/` directory:

```bash
agentform init
```

This clones the module repositories locally. You must run `agentform init` before compiling or running workflows that use external modules.

#### 3. Use module resources

Resources from modules are namespaced with `module.<name>`:

```hcl
workflow "review" {
  entry = step.start
  
  step "start" {
    type  = "llm"
    agent = agent.module.pr-reviewer.reviewer  // Use module's agent
    next  = step.end
  }
  
  step "end" { type = "end" }
}
```

Or run a module's workflow directly:

```bash
agentform run module.pr-reviewer.review_workflow .
```

### Creating Modules

A module is simply a directory containing `.af` files. To create a shareable module:

#### 1. Create the module structure

```
my-module/
├── 00-project.af      # Module metadata
├── 01-variables.af    # Input parameters (variables)
├── 02-providers.af    # LLM providers
├── 03-policies.af     # Policies
├── 04-models.af       # Model configurations
├── 05-agents.af       # Agent definitions
└── 06-workflows.af    # Workflows (optional)
```

#### 2. Define input variables

Variables without defaults become required parameters:

```hcl
// 01-variables.af
variable "api_key" {
  type        = string
  description = "API key for the LLM provider"
  sensitive   = true
  // No default = required parameter
}

variable "model" {
  type        = string
  description = "Model to use"
  default     = "gpt-4o-mini"  // Has default = optional
}
```

#### 3. Publish to Git

Push your module to a Git repository. Users can then reference it:

```hcl
module "my-module" {
  source  = "github.com/your-org/your-repo//path/to/module"
  version = "main"
  
  api_key = var.my_api_key
}
```

### Module Source Formats

| Format | Example |
|--------|---------|
| GitHub | `github.com/org/repo` |
| GitHub with subdirectory | `github.com/org/repo//modules/my-module` |
| GitLab | `gitlab.com/org/repo` |
| Local path | `./modules/my-module` |

### Module Caching

Modules are cached in `.af/modules/` within your project directory:

```
my-project/
├── .af/
│   └── modules/
│       └── github_com_org_repo_abc123/  # Cached module
├── 00-project.af
└── 01-modules.af
```

Add `.af/` to your `.gitignore` - these are downloaded dependencies.

<br />

## CLI Reference

```bash
# Initialize project - download external modules
agentform init [directory]

# Validate a specification
agentform validate <spec-file>

# Compile to IR (Intermediate Representation) - useful for debugging
agentform compile <spec-file> [--output output.json]

# Run a workflow
agentform run <workflow-name> [options]

Options:
  -s, --spec PATH        Path to .af spec file (default: agentform.af)
  -i, --input JSON       Input data as JSON string
  -f, --input-file PATH  Input data from JSON file
  -o, --output PATH      Write output to file
  -t, --trace PATH       Write execution trace to file
  -v, --verbose          Enable verbose output
```

<br />

---

<br />

## Contributing

We welcome contributions! Whether it's bug fixes, new features, or documentation improvements.

### Development Setup

#### Prerequisites

- Python 3.12 or higher
- [Poetry](https://python-poetry.org/) for dependency management

#### Clone and Install

```bash
# Clone the repository
git clone https://github.com/agentform-team/agentform.git
cd agentform

# Install all packages in development mode
cd agentform-schema && poetry install && cd ..
cd agentform-mcp && poetry install && cd ..
cd agentform-compiler && poetry install && cd ..
cd agentform-runtime && poetry install && cd ..
cd agentform-cli && poetry install && cd ..
```

#### Running Tests

```bash
# Run tests for a specific package
cd agentform-runtime
poetry run pytest

# Run with coverage
poetry run pytest --cov=agentform_runtime
```

### Project Structure

```
agentform/
├── agentform-schema/      # Core data models
├── agentform-compiler/    # Agentform parser and validator
├── agentform-runtime/     # Workflow execution engine
├── agentform-mcp/         # MCP client integration
├── agentform-cli/         # Command-line interface
└── examples/        # Example configurations (.af format)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

<br />

---

<p align="center">
  <sub>Built with ❤️ for the AI agent community</sub>
</p>


