# PR Reviewer Example

An ACP example demonstrating an automated pull request reviewer using the GitHub MCP server with human approval gates.

## Overview

This example showcases advanced ACP features including:
- **GitHub integration**: Fetch PR data and submit reviews via MCP
- **Human-in-the-loop**: Approval gates before submitting reviews
- **Write operations with approval**: Capabilities that require explicit approval
- **Multi-step data gathering**: Sequential capability calls to build context

## Prerequisites

1. OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

2. GitHub Personal Access Token with appropriate permissions:
   ```bash
   export GITHUB_PERSONAL_ACCESS_TOKEN="your-github-token"
   ```
   
   Required token scopes:
   - `repo` (for private repositories)
   - `public_repo` (for public repositories only)

3. Node.js and npm (for the MCP GitHub server):
   ```bash
   npm install -g @modelcontextprotocol/server-github
   ```

## Usage

Review a pull request:

```bash
acp run review_pr --spec spec.acp --input-file input.yaml
```

Or specify PR details inline:

```bash
acp run review_pr --spec spec.acp --input '{
  "owner": "myorg",
  "repo": "myrepo",
  "pr_number": 42
}'
```

The workflow will:
1. Fetch PR metadata
2. Fetch changed files
3. Generate a review using GPT-4
4. **Pause for your approval**
5. Submit the review to GitHub (only if approved)

## Spec File Structure

### Authenticated MCP Server
GitHub server requires authentication via environment variable.

```hcl
server "github" {
  type      = "mcp"
  transport = "stdio"
  command   = ["npx", "@modelcontextprotocol/server-github"]
  auth {
    token = env("GITHUB_PERSONAL_ACCESS_TOKEN")  // Auth token from env
  }
}
```

### Capabilities with Approval Requirements
Some capabilities modify state and require explicit human approval.

```hcl
// Read-only capabilities
capability "get_pr" {
  server      = server.github
  method      = "get_pull_request"
  side_effect = "read"
}

capability "list_pr_files" {
  server      = server.github
  method      = "get_pull_request_files"
  side_effect = "read"
}

// Write capability requiring approval
capability "create_review" {
  server           = server.github
  method           = "create_pull_request_review"
  side_effect      = "write"
  requires_approval = true  // Human must approve before execution
}
```

### Models and Expert Reviewer Agent
Specialized agent with detailed code review instructions.

```hcl
model "gpt4o" {
  provider = provider.llm.openai.default
  id       = "gpt-4o"
  params {
    temperature = 0.2  // Low temperature for consistent reviews
  }
}

agent "reviewer" {
  model = model.gpt4o  // Using capable model for code review

  instructions = <<EOF
You are an expert code reviewer. Review pull requests thoroughly.

Focus on:
- Code quality and best practices
- Potential bugs or edge cases
- Performance implications
- Security concerns
- Documentation and readability

Be constructive and specific in your feedback.
Suggest improvements where possible.
EOF

  allow  = [capability.get_pr, capability.list_pr_files, capability.create_review]
  policy = policy.review_policy
}
```

### Workflow with Human Approval
The workflow includes a human approval step before submitting reviews.

```hcl
workflow "review_pr" {
  entry = step.fetch_pr

  // Step 1: Fetch PR metadata
  step "fetch_pr" {
    type       = "call"
    capability = capability.get_pr

    args {
      owner       = input.owner
      repo        = input.repo
      pull_number = input.pr_number
    }

    output "pr_data" { from = result.data }

    next = step.fetch_files
  }

  // Step 2: Fetch changed files
  step "fetch_files" {
    type       = "call"
    capability = capability.list_pr_files

    args {
      owner       = input.owner
      repo        = input.repo
      pull_number = input.pr_number
    }

    output "pr_files" { from = result.data }

    next = step.analyze
  }

  // Step 3: Generate review with LLM
  step "analyze" {
    type  = "llm"
    agent = agent.reviewer

    input {
      pr    = state.pr_data
      files = state.pr_files
    }

    output "review" { from = result.text }

    next = step.approval
  }

  // Step 4: Human approval gate
  step "approval" {
    type      = "human_approval"
    payload   = "state.review"        // Show this to the human
    on_approve = step.submit_review   // If approved, continue
    on_reject  = step.end             // If rejected, stop
  }

  // Step 5: Submit the review
  step "submit_review" {
    type       = "call"
    capability = capability.create_review

    args {
      owner       = input.owner
      repo        = input.repo
      pull_number = input.pr_number
      body        = state.review
      event       = "COMMENT"  // COMMENT, APPROVE, or REQUEST_CHANGES
    }

    output "result" { from = result.data }

    next = step.end
  }

  step "end" { type = "end" }
}
```

## Input Schema

```json
{
  "owner": "organization-or-username",
  "repo": "repository-name",
  "pr_number": 123
}
```

## Key Concepts

### Human Approval Gates
The `human_approval` step type pauses workflow execution and waits for human input:

```hcl
step "approval" {
  type      = "human_approval"
  payload   = "state.review"      // Data to show the approver
  on_approve = step.submit_review  // Next step if approved
  on_reject  = step.end            // Next step if rejected
}
```

This ensures humans remain in control of consequential actions.

### Capability Approval
Capabilities marked with `requires_approval = true` trigger approval prompts even during LLM agent tool use:

```hcl
capability "create_review" {
  server           = server.github
  method           = "create_pull_request_review"
  side_effect      = "write"
  requires_approval = true
}
```

### Sequential Data Gathering
The workflow demonstrates gathering data across multiple capability calls before processing:
1. `fetch_pr` → gets PR title, description, author, etc.
2. `fetch_files` → gets list of changed files with diffs
3. `analyze` → LLM reviews with all context

### Review Event Types
When submitting reviews, the `event` parameter controls the review type:
- `COMMENT`: General feedback without approval/rejection
- `APPROVE`: Approve the PR
- `REQUEST_CHANGES`: Request changes before merge

## Safety Considerations

This example demonstrates multiple safety layers:

1. **Policy budgets**: Cost and time limits prevent runaway execution
2. **Capability approval**: Write operations require explicit approval
3. **Human approval step**: Final review before submitting to GitHub
4. **Read-first pattern**: Gather data with read operations, then review before writing

These patterns ensure the agent cannot take irreversible actions without human oversight.

