# Agentform CLI

Command line interface for Agentform.

## Installation

```bash
poetry install
```

## Usage

```bash
# Validate a spec
agentform validate my-agents.yaml

# Run a workflow
agentform run workflow_name --spec my-agents.yaml --input '{"key": "value"}'
```

