# ACP CLI

Command line interface for ACP.

## Installation

```bash
poetry install
```

## Usage

```bash
# Validate a spec
acp validate my-agents.yaml

# Run a workflow
acp run workflow_name --spec my-agents.yaml --input '{"key": "value"}'
```

