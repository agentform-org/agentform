"""Main Agentform client for programmatic workflow execution."""

from pathlib import Path
from typing import Any

from agentform_api.exceptions import CompilationError, WorkflowError
from agentform_api.types import WorkflowResult
from agentform_compiler import CompilationError as CompilerError
from agentform_compiler import compile_file
from agentform_runtime import ApprovalHandler, CLIApprovalHandler, WorkflowEngine
from agentform_runtime.engine import WorkflowError as RuntimeWorkflowError
from agentform_schema.ir import CompiledSpec


class Agentform:
    """Main Agentform client for programmatic workflow execution.

    This class provides a Python API for loading Agentform specifications and
    executing workflows. It wraps the agentform-compiler and agentform-runtime packages.

    Example:
        ```python
        from agentform_api import Agentform

        # Load from directory containing .agentform files
        agentform = Agentform.from_path("path/to/project/", variables={"api_key": "..."})

        # Run a workflow
        result = await agentform.run_workflow("my_workflow", input_data={"question": "Hello"})
        print(result.output)
        ```
    """

    def __init__(
        self,
        spec: CompiledSpec,
        approval_handler: ApprovalHandler | None = None,
        verbose: bool = False,
    ):
        """Initialize Agentform client with a compiled spec.

        Use `Agentform.from_path()` to create an instance from .agentform files.

        Args:
            spec: Compiled Agentform specification
            approval_handler: Custom handler for human approval steps (default: CLI)
            verbose: Enable verbose logging
        """
        self._spec = spec
        self._approval_handler = approval_handler or CLIApprovalHandler()
        self._verbose = verbose
        self._engine: WorkflowEngine | None = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        variables: dict[str, str] | None = None,
        approval_handler: ApprovalHandler | None = None,
        verbose: bool = False,
    ) -> "Agentform":
        """Create an Agentform instance from a file or directory path.

        If path is a directory, all .agentform files in it are discovered and compiled
        together (Terraform-style multi-file support).

        Args:
            path: Path to .agentform file or directory containing .agentform files
            variables: Variable values to substitute in the spec
            approval_handler: Custom handler for human approval steps (default: CLI)
            verbose: Enable verbose logging

        Returns:
            Agentform instance ready to run workflows

        Raises:
            CompilationError: If compilation fails

        Example:
            ```python
            # Load from directory
            agentform = Agentform.from_path("./my-project/", variables={"api_key": "sk-..."})

            # Load from single file
            agentform = Agentform.from_path("agent.agentform", variables={"api_key": "sk-..."})
            ```
        """
        path = Path(path)

        try:
            spec = compile_file(
                path,
                variables=variables,
                resolve_credentials=True,
            )
        except CompilerError as e:
            raise CompilationError(f"Failed to compile spec from {path}: {e}") from e

        return cls(spec, approval_handler=approval_handler, verbose=verbose)

    def _get_engine(self) -> WorkflowEngine:
        """Get or create the workflow engine."""
        if self._engine is None:
            self._engine = WorkflowEngine(
                self._spec,
                approval_handler=self._approval_handler,
                verbose=self._verbose,
            )
        return self._engine

    async def run_workflow(
        self,
        workflow_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Run a workflow by name.

        Args:
            workflow_name: Name of the workflow to execute
            input_data: Input data for the workflow (accessible via $input.field)

        Returns:
            WorkflowResult containing output, state, and execution trace

        Raises:
            WorkflowError: If workflow execution fails

        Example:
            ```python
            result = await agentform.run_workflow(
                "process", input_data={"question": "What is the capital of France?"}
            )
            print(result.output)  # Workflow output
            print(result.state)  # Full state including all step outputs
            ```
        """
        engine = self._get_engine()

        try:
            result = await engine.run(workflow_name, input_data)
            return WorkflowResult(
                output=result.get("output"),
                state=result.get("state", {}),
                trace=result.get("trace", {}),
            )
        except RuntimeWorkflowError as e:
            raise WorkflowError(f"Workflow '{workflow_name}' failed: {e}") from e
        except Exception as e:
            raise WorkflowError(f"Unexpected error in workflow '{workflow_name}': {e}") from e

    @property
    def workflows(self) -> list[str]:
        """Get list of available workflow names.

        Returns:
            List of workflow names defined in the spec
        """
        return list(self._spec.workflows.keys())

    @property
    def agents(self) -> list[str]:
        """Get list of available agent names.

        Returns:
            List of agent names defined in the spec
        """
        return list(self._spec.agents.keys())

    async def close(self) -> None:
        """Close the client and release resources.

        This stops any running MCP servers and cleans up the workflow engine.
        """
        if self._engine is not None:
            # The engine's MCP client will be closed when the engine is garbage collected
            # or when the context manager exits in the engine's run method
            self._engine = None

    async def __aenter__(self) -> "Agentform":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - cleanup resources."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        workflow_count = len(self._spec.workflows)
        agent_count = len(self._spec.agents)
        return f"<Agentform workflows={workflow_count} agents={agent_count}>"
