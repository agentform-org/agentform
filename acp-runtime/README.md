# ACP Runtime

Workflow execution engine for ACP.

## Installation

```bash
poetry install
```

## Usage

```python
from acp_runtime import WorkflowEngine

engine = WorkflowEngine(compiled_spec)
result = await engine.run("workflow_name", {"input": "data"})
```

