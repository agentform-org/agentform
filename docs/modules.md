---
layout: default
title: Modules
---

# Agentform Modules

Agentform supports a **Terraform-style module system** for creating reusable, shareable agent configurations. Modules let you package providers, policies, agents, and workflows together, making it easy for others to use without extensive configuration.

## Using Modules

### 1. Import a module in your project

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

### 2. Initialize your project

Download all external modules to your local `.agentform/modules/` directory:

```bash
agentform init
```

This clones the module repositories locally. You must run `agentform init` before compiling or running workflows that use external modules.

### 3. Use module resources

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

## Creating Modules

A module is simply a directory containing `.agentform` files. To create a shareable module:

### 1. Create the module structure

```
my-module/
├── 00-project.agentform      # Module metadata
├── 01-variables.agentform    # Input parameters (variables)
├── 02-providers.agentform    # LLM providers
├── 03-policies.agentform     # Policies
├── 04-models.agentform       # Model configurations
├── 05-agents.agentform       # Agent definitions
└── 06-workflows.agentform    # Workflows (optional)
```

### 2. Define input variables

Variables without defaults become required parameters:

```hcl
// 01-variables.agentform
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

### 3. Publish to Git

Push your module to a Git repository. Users can then reference it:

```hcl
module "my-module" {
  source  = "github.com/your-org/your-repo//path/to/module"
  version = "main"
  
  api_key = var.my_api_key
}
```

## Module Source Formats

| Format | Example |
|--------|---------|
| GitHub | `github.com/org/repo` |
| GitHub with subdirectory | `github.com/org/repo//modules/my-module` |
| GitLab | `gitlab.com/org/repo` |
| Local path | `./modules/my-module` |

## Module Caching

Modules are cached in `.agentform/modules/` within your project directory:

```
my-project/
├── .agentform/
│   └── modules/
│       └── github_com_org_repo_abc123/  # Cached module
├── 00-project.agentform
└── 01-modules.agentform
```

Add `.agentform/` to your `.gitignore` - these are downloaded dependencies.

## Best Practices

1. **Version your modules**: Use Git tags for stable releases
2. **Document variables**: Provide clear descriptions for all variables
3. **Use sensible defaults**: Make modules easy to use out of the box
4. **Test modules**: Ensure modules work before publishing
5. **Follow naming conventions**: Use descriptive names for resources
