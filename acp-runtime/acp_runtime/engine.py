"""Workflow execution engine."""

from typing import Any

from acp_mcp import MCPClient
from acp_schema.ir import CompiledSpec, ResolvedStep
from acp_schema.models import StepType

from acp_runtime.approval import ApprovalHandler, CLIApprovalHandler
from acp_runtime.llm import LLMExecutor
from acp_runtime.policy import PolicyEnforcer
from acp_runtime.state import WorkflowState
from acp_runtime.tracing import Tracer


class WorkflowError(Exception):
    """Error during workflow execution."""

    pass


class WorkflowEngine:
    """Executes workflows defined in compiled specs."""

    def __init__(
        self,
        spec: CompiledSpec,
        approval_handler: ApprovalHandler | None = None,
    ):
        """Initialize workflow engine.

        Args:
            spec: Compiled specification (IR)
            approval_handler: Handler for human approvals (default: CLI)
        """
        self._spec = spec
        self._approval_handler = approval_handler or CLIApprovalHandler()
        self._llm_executor = LLMExecutor(spec.providers)
        self._policy_enforcer = PolicyEnforcer(spec.policies)
        self._mcp_client: MCPClient | None = None

    async def _init_mcp(self) -> MCPClient:
        """Initialize MCP client with all servers."""
        if self._mcp_client is not None:
            return self._mcp_client

        client = MCPClient()
        for server_name, server in self._spec.servers.items():
            auth_token = None
            if server.auth_token and server.auth_token.value:
                auth_token = server.auth_token.value
            client.add_server(server_name, server.command, auth_token)

        await client.start_all()
        self._mcp_client = client
        return client

    async def _close_mcp(self) -> None:
        """Close MCP client."""
        if self._mcp_client:
            await self._mcp_client.stop_all()
            self._mcp_client = None

    async def run(
        self,
        workflow_name: str,
        input_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a workflow.

        Args:
            workflow_name: Name of workflow to run
            input_data: Input data for the workflow

        Returns:
            Dict with 'output', 'state', and 'trace' keys

        Raises:
            WorkflowError: If workflow execution fails
        """
        workflow = self._spec.workflows.get(workflow_name)
        if not workflow:
            raise WorkflowError(f"Workflow '{workflow_name}' not found")

        # Initialize
        state = WorkflowState(input_data)
        tracer = Tracer(workflow_name)
        context_id = tracer.trace_id

        self._policy_enforcer.start_context(context_id)
        tracer.workflow_start(input_data or {})

        try:
            # Initialize MCP if we have servers
            if self._spec.servers:
                await self._init_mcp()

            # Execute steps
            current_step_id = workflow.entry_step
            final_output = None

            while current_step_id:
                step = workflow.steps.get(current_step_id)
                if not step:
                    raise WorkflowError(f"Step '{current_step_id}' not found")

                tracer.step_start(step.id, step.type.value)

                try:
                    result, next_step = await self._execute_step(
                        step, state, tracer, context_id
                    )

                    if step.save_as:
                        state.set(step.save_as, result)

                    tracer.step_end(step.id, result)
                    final_output = result
                    current_step_id = next_step

                except Exception as e:
                    tracer.step_error(step.id, e)
                    raise

            tracer.workflow_end(final_output)

            return {
                "output": final_output,
                "state": state.to_dict(),
                "trace": tracer.to_json(),
            }

        except Exception as e:
            tracer.workflow_error(e)
            raise WorkflowError(f"Workflow execution failed: {e}") from e

        finally:
            self._policy_enforcer.end_context(context_id)
            await self._close_mcp()

    async def _execute_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
        tracer: Tracer,
        context_id: str,
    ) -> tuple[Any, str | None]:
        """Execute a single workflow step.

        Args:
            step: Step to execute
            state: Current workflow state
            tracer: Tracer for recording events
            context_id: Policy context ID

        Returns:
            Tuple of (result, next_step_id)
        """
        if step.type == StepType.END:
            return None, None

        elif step.type == StepType.LLM:
            return await self._execute_llm_step(step, state, tracer, context_id)

        elif step.type == StepType.CALL:
            return await self._execute_call_step(step, state, tracer, context_id)

        elif step.type == StepType.CONDITION:
            return await self._execute_condition_step(step, state)

        elif step.type == StepType.HUMAN_APPROVAL:
            return await self._execute_approval_step(step, state, tracer)

        else:
            raise WorkflowError(f"Unknown step type: {step.type}")

    async def _execute_llm_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
        tracer: Tracer,
        context_id: str,
    ) -> tuple[Any, str | None]:
        """Execute an LLM step."""
        if not step.agent_name:
            raise WorkflowError(f"LLM step '{step.id}' missing agent")

        agent = self._spec.agents.get(step.agent_name)
        if not agent:
            raise WorkflowError(f"Agent '{step.agent_name}' not found")

        # Check policy
        self._policy_enforcer.check_timeout(context_id, agent.policy_name)

        # Resolve input
        input_data = {}
        if step.input_mapping:
            input_data = state.resolve_dict(step.input_mapping)

        # Execute LLM
        result = await self._llm_executor.execute(agent, input_data)

        tracer.llm_call(
            step.id,
            result.get("model", "unknown"),
            str(input_data),
            str(result.get("response", "")),
        )

        return result, step.next_step

    async def _execute_call_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
        tracer: Tracer,
        context_id: str,
    ) -> tuple[Any, str | None]:
        """Execute a capability call step."""
        if not step.capability_name:
            raise WorkflowError(f"Call step '{step.id}' missing capability")

        capability = self._spec.capabilities.get(step.capability_name)
        if not capability:
            raise WorkflowError(f"Capability '{step.capability_name}' not found")

        # Find agent policy for this capability
        policy_name = None
        for agent in self._spec.agents.values():
            if step.capability_name in agent.allowed_capabilities:
                policy_name = agent.policy_name
                break

        # Check policy
        self._policy_enforcer.check_before_capability_call(context_id, policy_name)

        # Resolve args
        args = {}
        if step.args_mapping:
            args = state.resolve_dict(step.args_mapping)

        # Check if approval is required
        if capability.requires_approval:
            approved = await self._approval_handler.request_approval(
                tracer.workflow_name,
                step.id,
                {"capability": step.capability_name, "args": args},
            )
            if not approved:
                return {"approved": False, "skipped": True}, step.next_step

        # Execute capability
        if not self._mcp_client:
            raise WorkflowError("MCP client not initialized")

        result = await self._mcp_client.call_tool(
            capability.server_name,
            capability.method_name,
            args,
        )

        self._policy_enforcer.record_capability_call(context_id)

        tracer.capability_call(step.id, step.capability_name, args, result)

        return result, step.next_step

    async def _execute_condition_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
    ) -> tuple[Any, str | None]:
        """Execute a condition step."""
        if not step.condition_expr:
            raise WorkflowError(f"Condition step '{step.id}' missing condition")

        # Simple condition evaluation
        # Supports: $state.x == "value", $state.x, !$state.x
        condition = step.condition_expr.strip()

        if "==" in condition:
            parts = condition.split("==")
            left = state.resolve(parts[0].strip())
            right = parts[1].strip().strip('"').strip("'")
            result = str(left) == right
        elif "!=" in condition:
            parts = condition.split("!=")
            left = state.resolve(parts[0].strip())
            right = parts[1].strip().strip('"').strip("'")
            result = str(left) != right
        elif condition.startswith("!"):
            result = not bool(state.resolve(condition[1:]))
        else:
            result = bool(state.resolve(condition))

        next_step = step.on_true_step if result else step.on_false_step
        return {"condition": result}, next_step

    async def _execute_approval_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
        tracer: Tracer,
    ) -> tuple[Any, str | None]:
        """Execute a human approval step."""
        # Resolve payload
        payload = None
        if step.payload_expr:
            payload = state.resolve(step.payload_expr)

        tracer.approval_request(step.id, payload)

        approved = await self._approval_handler.request_approval(
            tracer.workflow_name,
            step.id,
            payload,
        )

        tracer.approval_response(step.id, approved)

        next_step = step.on_approve_step if approved else step.on_reject_step
        return {"approved": approved}, next_step

