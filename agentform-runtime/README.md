# Agentform Runtime

Workflow execution engine for Agentform.

## Installation

```bash
poetry install
```

## Usage

```python
from agentform_runtime import WorkflowEngine

engine = WorkflowEngine(compiled_spec)
result = await engine.run("workflow_name", {"input": "data"})
```

