# ACP API

Python SDK for programmatic ACP usage. Use ACP workflows directly from your Python applications.

## Installation

```bash
pip install acp-api
```

## Quick Start

```python
import asyncio
from acp_api import ACP

async def main():
    # Load from directory containing .acp files
    acp = ACP.from_path(
        "path/to/project/",
        variables={"openai_api_key": "sk-..."}
    )

    # Run a workflow
    result = await acp.run_workflow(
        "ask",
        input_data={"question": "What is the capital of France?"}
    )

    print(result.output)

asyncio.run(main())
```

## Usage

### Loading Specs

Load from a directory (discovers all `.acp` files):

```python
acp = ACP.from_path("./my-project/", variables={"api_key": "..."})
```

Load from a single file:

```python
acp = ACP.from_path("agent.acp", variables={"api_key": "..."})
```

### Running Workflows

```python
result = await acp.run_workflow("workflow_name", input_data={"key": "value"})

# Access results
print(result.output)  # Workflow output
print(result.state)   # Full state with all step outputs
print(result.trace)   # Execution trace for debugging
```

### Listing Available Workflows and Agents

```python
acp = ACP.from_path("./project/", variables={...})

print(acp.workflows)  # ['ask', 'process', ...]
print(acp.agents)     # ['assistant', 'reviewer', ...]
```

### Using Async Context Manager

For automatic resource cleanup:

```python
async with ACP.from_path("./project/", variables={...}) as acp:
    result = await acp.run_workflow("workflow", input_data={...})
    # Resources automatically cleaned up on exit
```

### Custom Approval Handlers

For workflows with human approval steps, provide a custom handler:

```python
from acp_runtime import ApprovalHandler

class SlackApprovalHandler(ApprovalHandler):
    async def request_approval(self, payload: dict) -> bool:
        # Send to Slack and wait for response
        return await send_slack_approval(payload)

acp = ACP.from_path(
    "./project/",
    variables={...},
    approval_handler=SlackApprovalHandler()
)
```

### Error Handling

```python
from acp_api import ACP, CompilationError, WorkflowError

try:
    acp = ACP.from_path("./project/", variables={...})
    result = await acp.run_workflow("workflow", input_data={...})
except CompilationError as e:
    print(f"Failed to compile spec: {e}")
except WorkflowError as e:
    print(f"Workflow failed: {e}")
```

## Integration Examples

### FastAPI

```python
from fastapi import FastAPI
from acp_api import ACP

app = FastAPI()
acp = ACP.from_path("./agents/", variables={"api_key": os.getenv("API_KEY")})

@app.post("/chat")
async def chat(message: str):
    result = await acp.run_workflow("chat", input_data={"message": message})
    return {"response": result.output}
```

### Background Tasks (Celery)

```python
from celery import Celery
from acp_api import ACP

celery = Celery()
acp = ACP.from_path("./workflows/", variables={...})

@celery.task
async def process_document(doc_id: str):
    result = await acp.run_workflow("process", input_data={"doc_id": doc_id})
    return result.output
```

### Testing

```python
import pytest
from acp_api import ACP

@pytest.fixture
def acp():
    return ACP.from_path("./test-specs/", variables={"api_key": "test"})

async def test_workflow(acp):
    result = await acp.run_workflow("test", input_data={"input": "test"})
    assert result.output is not None
```

## API Reference

### `ACP`

Main client class.

#### `ACP.from_path(path, variables=None, approval_handler=None, verbose=False)`

Create an ACP instance from a file or directory path.

- `path`: Path to `.acp` file or directory containing `.acp` files
- `variables`: Dictionary of variable values to substitute
- `approval_handler`: Custom handler for human approval steps
- `verbose`: Enable verbose logging

#### `await acp.run_workflow(workflow_name, input_data=None)`

Run a workflow by name.

- `workflow_name`: Name of the workflow to execute
- `input_data`: Input data for the workflow

Returns `WorkflowResult`.

#### `acp.workflows`

List of available workflow names.

#### `acp.agents`

List of available agent names.

#### `await acp.close()`

Close the client and release resources.

### `WorkflowResult`

Result of a workflow execution.

- `output`: The final output from the workflow
- `state`: Full workflow state dictionary
- `trace`: Execution trace for debugging

### Exceptions

- `ACPError`: Base exception for ACP API errors
- `CompilationError`: Error during ACP spec compilation
- `WorkflowError`: Error during workflow execution
