# PR Reviewer Example

An ACP example demonstrating an automated pull request reviewer using the GitHub MCP server with human approval gates.

## Overview

This example showcases advanced ACP features including:
- **GitHub integration**: Fetch PR data and submit reviews via MCP
- **Human-in-the-loop**: Approval gates before submitting reviews
- **Write operations with approval**: Capabilities that require explicit approval
- **Multi-step data gathering**: Sequential capability calls to build context

## File Structure

```
pr-reviewer/
├── 00-project.acp       # Project metadata (acp block)
├── 01-variables.acp     # Variable definitions
├── 02-providers.acp     # Provider and model definitions
├── 03-servers.acp       # MCP server configuration
├── 04-capabilities.acp  # Capability definitions
├── 05-policies.acp      # Policy definitions
├── 06-agents.acp        # Agent definitions
├── 07-workflows.acp     # Workflow definitions
├── input.yaml           # Sample input
└── README.md
```

## Prerequisites

1. OpenAI API key

2. GitHub Personal Access Token with scopes:
   - `repo` (for private repositories)
   - `public_repo` (for public repositories only)

3. Node.js and npm (for the MCP GitHub server):
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Usage

Run from the example directory:

```bash
cd examples/pr-reviewer

# Review a pull request
acp run review_pr \
  --var openai_api_key=$OPENAI_API_KEY \
  --var github_personal_access_token=$GITHUB_TOKEN \
  --input-file input.yaml

# Or specify PR details inline
acp run review_pr \
  --var openai_api_key=$OPENAI_API_KEY \
  --var github_personal_access_token=$GITHUB_TOKEN \
  --input '{"owner": "myorg", "repo": "myrepo", "pr_number": 42}'
```

The workflow will:
1. Fetch PR metadata
2. Fetch changed files
3. Generate a review using GPT-4
4. **Pause for your approval**
5. Submit the review to GitHub (only if approved)

To validate:

```bash
acp validate --var openai_api_key=test --var github_personal_access_token=test
```

## Key Concepts

### Variables (`01-variables.acp`)

Sensitive credentials are defined as variables:

```hcl
variable "openai_api_key" {
  type        = string
  sensitive   = true
}

variable "github_personal_access_token" {
  type        = string
  sensitive   = true
}
```

### Authenticated MCP Server (`03-servers.acp`)

```hcl
server "github" {
  type      = "mcp"
  transport = "stdio"
  command   = ["npx", "@modelcontextprotocol/server-github"]
  auth {
    token = var.github_personal_access_token
  }
}
```

### Capabilities with Approval (`04-capabilities.acp`)

```hcl
capability "create_review" {
  server           = server.github
  method           = "create_pull_request_review"
  side_effect      = "write"
  requires_approval = true  # Human must approve
}
```

### Human Approval Gates (`07-workflows.acp`)

```hcl
step "approval" {
  type    = "human_approval"
  payload = state.review
  on_approve = step.submit_review
  on_reject  = step.end
}
```

### Sequential Data Gathering

1. `fetch_pr` → gets PR title, description, author
2. `fetch_files` → gets list of changed files with diffs
3. `analyze` → LLM reviews with all context

## Input Schema

```json
{
  "owner": "organization-or-username",
  "repo": "repository-name",
  "pr_number": 123
}
```

## Safety Considerations

Multiple safety layers:
1. **Policy budgets**: Cost and time limits
2. **Capability approval**: Write operations require explicit approval
3. **Human approval step**: Final review before submitting to GitHub
4. **Read-first pattern**: Gather data before any write operations
