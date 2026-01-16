"""Workflow execution engine."""

from typing import Any

from agentform_mcp import MCPClient
from agentform_runtime.approval import ApprovalHandler, CLIApprovalHandler
from agentform_runtime.llm import LLMExecutor
from agentform_runtime.logging_config import get_logger
from agentform_runtime.policy import PolicyEnforcer
from agentform_runtime.state import WorkflowState
from agentform_runtime.tracing import Tracer
from agentform_schema.ir import CompiledSpec, ResolvedStep
from agentform_schema.models import StepType


class WorkflowError(Exception):
    """Error during workflow execution."""

    pass


class WorkflowEngine:
    """Executes workflows defined in compiled specs."""

    def __init__(
        self,
        spec: CompiledSpec,
        approval_handler: ApprovalHandler | None = None,
        verbose: bool = False,
    ):
        """Initialize workflow engine.

        Args:
            spec: Compiled specification (IR)
            approval_handler: Handler for human approvals (default: CLI)
            verbose: Enable verbose logging
        """
        self._spec = spec
        self._approval_handler = approval_handler or CLIApprovalHandler()
        self._llm_executor = LLMExecutor(spec.providers, verbose=verbose)
        self._policy_enforcer = PolicyEnforcer(spec.policies)
        self._mcp_client: MCPClient | None = None
        self._verbose = verbose
        self._logger = get_logger("agentform_runtime.engine")

    async def _init_mcp(self) -> MCPClient:
        """Initialize MCP client with all servers."""
        if self._mcp_client is not None:
            return self._mcp_client

        self._logger.info("mcp_initialization_start", server_count=len(self._spec.servers))
        client = MCPClient()
        for server_name, server in self._spec.servers.items():
            auth_token = None
            if server.auth_token and server.auth_token.value:
                auth_token = server.auth_token.value
            client.add_server(server_name, server.command, auth_token)
            self._logger.debug(
                "mcp_server_added", server_name=server_name, has_auth=bool(auth_token)
            )

        await client.start_all()
        self._mcp_client = client
        self._logger.info("mcp_initialization_complete", server_count=len(self._spec.servers))
        return client

    async def _close_mcp(self) -> None:
        """Close MCP client."""
        if self._mcp_client:
            self._logger.info("mcp_shutdown_start")
            await self._mcp_client.stop_all()
            self._mcp_client = None
            self._logger.info("mcp_shutdown_complete")

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

        self._logger.info(
            "workflow_initialization",
            workflow_name=workflow_name,
            trace_id=context_id,
            input_keys=list(input_data.keys()) if input_data else [],
        )

        self._policy_enforcer.start_context(context_id)
        tracer.workflow_start(input_data or {})

        try:
            # Initialize MCP if we have servers
            if self._spec.servers:
                await self._init_mcp()
            else:
                self._logger.debug("no_mcp_servers", workflow_name=workflow_name)

            # Execute steps
            current_step_id: str | None = workflow.entry_step
            final_output = None
            step_count = 0

            self._logger.info(
                "workflow_execution_start", workflow_name=workflow_name, entry_step=current_step_id
            )

            while current_step_id:
                step = workflow.steps.get(current_step_id)
                if not step:
                    self._logger.error(
                        "step_not_found", step_id=current_step_id, workflow_name=workflow_name
                    )
                    raise WorkflowError(f"Step '{current_step_id}' not found")

                step_count += 1
                self._logger.info(
                    "step_execution_start",
                    step_id=step.id,
                    step_type=step.type.value,
                    step_number=step_count,
                    workflow_name=workflow_name,
                )

                tracer.step_start(step.id, step.type.value)

                try:
                    result, next_step = await self._execute_step(step, state, tracer, context_id)

                    if step.save_as:
                        state.set(step.save_as, result)
                        self._logger.debug(
                            "state_saved",
                            step_id=step.id,
                            save_as=step.save_as,
                            has_result=result is not None,
                        )

                    tracer.step_end(step.id, result)
                    self._logger.info(
                        "step_execution_complete",
                        step_id=step.id,
                        step_type=step.type.value,
                        next_step=next_step,
                        has_result=result is not None,
                    )
                    final_output = result
                    current_step_id = next_step

                except Exception as e:
                    self._logger.error(
                        "step_execution_error",
                        step_id=step.id,
                        step_type=step.type.value,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    tracer.step_error(step.id, e)
                    raise

            self._logger.info(
                "workflow_execution_complete",
                workflow_name=workflow_name,
                total_steps=step_count,
                has_output=final_output is not None,
            )
            tracer.workflow_end(final_output)

            return {
                "output": final_output,
                "state": state.to_dict(),
                "trace": tracer.to_json(),
            }

        except Exception as e:
            self._logger.error(
                "workflow_execution_error",
                workflow_name=workflow_name,
                error=str(e),
                error_type=type(e).__name__,
            )
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

        self._logger.info(
            "llm_step_start",
            step_id=step.id,
            agent_name=step.agent_name,
            provider=agent.provider_name,
            model_preference=agent.model_preference,
            model_fallback=agent.model_fallback,
        )

        # Check policy
        self._policy_enforcer.check_timeout(context_id, agent.policy_name)
        self._logger.debug("policy_check_passed", step_id=step.id, policy_name=agent.policy_name)

        # Resolve input
        input_data = {}
        if step.input_mapping:
            input_data = state.resolve_dict(step.input_mapping)

        self._logger.debug(
            "llm_input_resolved", step_id=step.id, input_keys=list(input_data.keys())
        )

        # Execute LLM
        result = await self._llm_executor.execute(agent, input_data)

        self._logger.info(
            "llm_step_complete",
            step_id=step.id,
            agent_name=step.agent_name,
            model=result.get("model", "unknown"),
            provider=result.get("provider", "unknown"),
            usage=result.get("usage"),
        )

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

        self._logger.info(
            "call_step_start",
            step_id=step.id,
            capability_name=step.capability_name,
            server_name=capability.server_name,
            method=capability.method_name,
            requires_approval=capability.requires_approval,
        )

        # Find agent policy for this capability
        policy_name = None
        for agent in self._spec.agents.values():
            if step.capability_name in agent.allowed_capabilities:
                policy_name = agent.policy_name
                break

        # Check policy
        self._policy_enforcer.check_before_capability_call(context_id, policy_name)
        self._logger.debug("policy_check_passed", step_id=step.id, policy_name=policy_name)

        # Resolve args
        args = {}
        if step.args_mapping:
            args = state.resolve_dict(step.args_mapping)

        self._logger.debug("call_args_resolved", step_id=step.id, args_keys=list(args.keys()))

        # Check if approval is required
        if capability.requires_approval:
            self._logger.info("approval_required", step_id=step.id, capability=step.capability_name)
            approved = await self._approval_handler.request_approval(
                tracer.workflow_name,
                step.id,
                {"capability": step.capability_name, "args": args},
            )
            self._logger.info("approval_result", step_id=step.id, approved=approved)
            if not approved:
                self._logger.warning(
                    "capability_call_skipped", step_id=step.id, reason="not_approved"
                )
                return {"approved": False, "skipped": True}, step.next_step

        # Execute capability
        if not self._mcp_client:
            raise WorkflowError("MCP client not initialized")

        self._logger.debug(
            "mcp_call_start",
            step_id=step.id,
            server=capability.server_name,
            method=capability.method_name,
        )
        result = await self._mcp_client.call_tool(
            capability.server_name,
            capability.method_name,
            args,
        )
        self._logger.info("mcp_call_complete", step_id=step.id, has_result=result is not None)

        self._policy_enforcer.record_capability_call(context_id)

        tracer.capability_call(step.id, step.capability_name, args, result)

        return result, step.next_step

    async def _execute_condition_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
    ) -> tuple[Any, str | None]:
        """Execute a condition step.

        Supports:
        - Simple comparisons: $state.x == "value", $state.x != "value"
        - Boolean checks: $state.x, !$state.x
        - Logical operators: $state.x && $state.y, $state.x || $state.y
        - Conditional expressions: condition ? true_val : false_val
        - Comparison operators: <, >, <=, >=
        """
        if not step.condition_expr:
            raise WorkflowError(f"Condition step '{step.id}' missing condition")

        self._logger.info("condition_step_start", step_id=step.id, condition=step.condition_expr)

        # Use the enhanced evaluate_condition method from WorkflowState
        result = state.evaluate_condition(step.condition_expr)

        next_step = step.on_true_step if result else step.on_false_step
        self._logger.info(
            "condition_step_complete",
            step_id=step.id,
            condition_result=result,
            next_step=next_step,
            branch="true" if result else "false",
        )
        return {"condition": result}, next_step

    async def _execute_approval_step(
        self,
        step: ResolvedStep,
        state: WorkflowState,
        tracer: Tracer,
    ) -> tuple[Any, str | None]:
        """Execute a human approval step."""
        self._logger.info("approval_step_start", step_id=step.id)

        # Resolve payload
        payload = None
        if step.payload_expr:
            payload = state.resolve(step.payload_expr)
            self._logger.debug(
                "approval_payload_resolved", step_id=step.id, has_payload=payload is not None
            )

        tracer.approval_request(step.id, payload)

        approved = await self._approval_handler.request_approval(
            tracer.workflow_name,
            step.id,
            payload,
        )

        self._logger.info("approval_step_complete", step_id=step.id, approved=approved)
        tracer.approval_response(step.id, approved)

        next_step = step.on_approve_step if approved else step.on_reject_step
        return {"approved": approved}, next_step
