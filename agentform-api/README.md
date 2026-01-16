# Agentform API

Python SDK for programmatic Agentform usage. Use Agentform workflows directly from your Python applications.

## Installation

```bash
pip install agentform-api
```

## Quick Start

```python
import asyncio
from agentform_api import Agentform

async def main():
    # Load from directory containing .agentform files
    agentform = Agentform.from_path(
        "path/to/project/",
        variables={"openai_api_key": "sk-..."}
    )

    # Run a workflow
    result = await agentform.run_workflow(
        "ask",
        input_data={"question": "What is the capital of France?"}
    )

    print(result.output)

asyncio.run(main())
```

## Usage

### Loading Specs

Load from a directory (discovers all `.agentform` files):

```python
agentform = Agentform.from_path("./my-project/", variables={"api_key": "..."})
```

Load from a single file:

```python
agentform = Agentform.from_path("agent.agentform", variables={"api_key": "..."})
```

### Running Workflows

```python
result = await agentform.run_workflow("workflow_name", input_data={"key": "value"})

# Access results
print(result.output)  # Workflow output
print(result.state)   # Full state with all step outputs
print(result.trace)   # Execution trace for debugging
```

### Listing Available Workflows and Agents

```python
agentform = Agentform.from_path("./project/", variables={...})

print(agentform.workflows)  # ['ask', 'process', ...]
print(agentform.agents)     # ['assistant', 'reviewer', ...]
```

### Using Async Context Manager

For automatic resource cleanup:

```python
async with Agentform.from_path("./project/", variables={...}) as agentform:
    result = await agentform.run_workflow("workflow", input_data={...})
    # Resources automatically cleaned up on exit
```

### Custom Approval Handlers

For workflows with human approval steps, provide a custom handler:

```python
from agentform_runtime import ApprovalHandler

class SlackApprovalHandler(ApprovalHandler):
    async def request_approval(self, payload: dict) -> bool:
        # Send to Slack and wait for response
        return await send_slack_approval(payload)

agentform = Agentform.from_path(
    "./project/",
    variables={...},
    approval_handler=SlackApprovalHandler()
)
```

### Error Handling

```python
from agentform_api import Agentform, CompilationError, WorkflowError

try:
    agentform = Agentform.from_path("./project/", variables={...})
    result = await agentform.run_workflow("workflow", input_data={...})
except CompilationError as e:
    print(f"Failed to compile spec: {e}")
except WorkflowError as e:
    print(f"Workflow failed: {e}")
```

## Integration Examples

### FastAPI

```python
from fastapi import FastAPI
from agentform_api import Agentform

app = FastAPI()
agentform = Agentform.from_path("./agents/", variables={"api_key": os.getenv("API_KEY")})

@app.post("/chat")
async def chat(message: str):
    result = await agentform.run_workflow("chat", input_data={"message": message})
    return {"response": result.output}
```

### Background Tasks (Celery)

```python
from celery import Celery
from agentform_api import Agentform

celery = Celery()
agentform = Agentform.from_path("./workflows/", variables={...})

@celery.task
async def process_document(doc_id: str):
    result = await agentform.run_workflow("process", input_data={"doc_id": doc_id})
    return result.output
```

### Testing

```python
import pytest
from agentform_api import Agentform

@pytest.fixture
def agentform():
    return Agentform.from_path("./test-specs/", variables={"api_key": "test"})

async def test_workflow(agentform):
    result = await agentform.run_workflow("test", input_data={"input": "test"})
    assert result.output is not None
```

## API Reference

### `Agentform`

Main client class.

#### `Agentform.from_path(path, variables=None, approval_handler=None, verbose=False)`

Create an Agentform instance from a file or directory path.

- `path`: Path to `.agentform` file or directory containing `.agentform` files
- `variables`: Dictionary of variable values to substitute
- `approval_handler`: Custom handler for human approval steps
- `verbose`: Enable verbose logging

#### `await agentform.run_workflow(workflow_name, input_data=None)`

Run a workflow by name.

- `workflow_name`: Name of the workflow to execute
- `input_data`: Input data for the workflow

Returns `WorkflowResult`.

#### `agentform.workflows`

List of available workflow names.

#### `agentform.agents`

List of available agent names.

#### `await agentform.close()`

Close the client and release resources.

### `WorkflowResult`

Result of a workflow execution.

- `output`: The final output from the workflow
- `state`: Full workflow state dictionary
- `trace`: Execution trace for debugging

### Exceptions

- `AgentformError`: Base exception for Agentform API errors
- `CompilationError`: Error during Agentform spec compilation
- `WorkflowError`: Error during workflow execution
