---
layout: default
title: CLI Reference
---

# CLI Reference

Complete reference for the `agentform` command-line interface.

## Commands

### `agentform init`

Initialize a project and download external modules.

```bash
agentform init [directory]
```

**Description:** Downloads all external modules referenced in your `.agentform` files to the local `.agentform/modules/` directory. You must run this before compiling or running workflows that use external modules.

**Arguments:**
- `directory` (optional): Project directory. Defaults to current directory.

**Example:**
```bash
agentform init
agentform init ./my-project
```

---

### `agentform validate`

Validate a specification file.

```bash
agentform validate <spec-file> [options]
```

**Description:** Checks that your Agentform configuration is valid, all references are correct, and all required variables are provided.

**Arguments:**
- `spec-file`: Path to `.agentform` spec file or directory containing `.agentform` files

**Options:**
- `-v, --var KEY=VALUE`: Set a variable value (can be used multiple times)
- `--var-file PATH`: Load variables from a JSON file

**Example:**
```bash
agentform validate my-agent.agentform
agentform validate . --var openai_api_key=$OPENAI_API_KEY
```

---

### `agentform compile`

Compile a specification to Intermediate Representation (IR).

```bash
agentform compile <spec-file> [options]
```

**Description:** Parses and validates your `.agentform` files and generates the Intermediate Representation (IR) JSON. Useful for debugging and understanding how Agentform interprets your configuration.

**Arguments:**
- `spec-file`: Path to `.agentform` spec file or directory containing `.agentform` files

**Options:**
- `-o, --output PATH`: Write IR to file (default: stdout)
- `-v, --var KEY=VALUE`: Set a variable value (can be used multiple times)
- `--var-file PATH`: Load variables from a JSON file

**Example:**
```bash
agentform compile my-agent.agentform
agentform compile . --output ir.json --var openai_api_key=$OPENAI_API_KEY
```

---

### `agentform run`

Run a workflow.

```bash
agentform run <workflow-name> [options]
```

**Description:** Executes a workflow from your Agentform specification. The workflow name can be a simple name (e.g., `ask`) or a module-qualified name (e.g., `module.my-module.workflow_name`).

**Arguments:**
- `workflow-name`: Name of the workflow to run

**Options:**
- `-s, --spec PATH`: Path to `.agentform` spec file or directory (default: `agentform.agentform` or current directory)
- `-i, --input JSON`: Input data as JSON string
- `-f, --input-file PATH`: Input data from JSON file
- `-o, --output PATH`: Write output to file (default: stdout)
- `-t, --trace PATH`: Write execution trace to file
- `-v, --var KEY=VALUE`: Set a variable value (can be used multiple times)
- `--var-file PATH`: Load variables from a JSON file
- `--verbose`: Enable verbose output

**Example:**
```bash
# Simple workflow
agentform run ask --spec my-agent.agentform --input '{"question": "Hello!"}'

# With input file
agentform run ask --input-file input.json

# With variables
agentform run ask --var openai_api_key=$OPENAI_API_KEY --input '{"question": "Hello!"}'

# Save trace for debugging
agentform run ask --trace trace.json --input '{"question": "Hello!"}'

# Module workflow
agentform run module.pr-reviewer.review_workflow .
```

---

## Global Options

These options are available for all commands:

- `-h, --help`: Show help message
- `--version`: Show version information

---

## Environment Variables

You can also set variables using environment variables with the `AGENTFORM_VAR_` prefix:

```bash
export AGENTFORM_VAR_openai_api_key="your-key"
agentform run ask --input '{"question": "Hello!"}'
```

---

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Validation error
- `3`: Compilation error
- `4`: Runtime error

---

## Examples

### Complete workflow

```bash
# 1. Initialize project (download modules)
agentform init

# 2. Validate configuration
agentform validate . --var openai_api_key=$OPENAI_API_KEY

# 3. Run workflow
agentform run ask \
  --var openai_api_key=$OPENAI_API_KEY \
  --input-file input.json \
  --output result.json \
  --trace trace.json
```

### Debugging

```bash
# Compile to see IR
agentform compile . --output ir.json --var openai_api_key=$OPENAI_API_KEY

# Run with verbose output
agentform run ask --verbose --input '{"question": "Hello!"}'

# Save trace for analysis
agentform run ask --trace trace.json --input '{"question": "Hello!"}'
```
