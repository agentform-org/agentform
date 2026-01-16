"""Human approval handling for workflow execution."""

import json
from abc import ABC, abstractmethod
from typing import Any


class ApprovalHandler(ABC):
    """Abstract base class for approval handlers."""

    @abstractmethod
    async def request_approval(
        self,
        workflow_name: str,
        step_id: str,
        payload: Any,
    ) -> bool:
        """Request approval for a step.

        Args:
            workflow_name: Name of the workflow
            step_id: ID of the step requiring approval
            payload: Data to show to approver

        Returns:
            True if approved, False if rejected
        """
        pass


class CLIApprovalHandler(ApprovalHandler):
    """Approval handler that prompts via CLI."""

    async def request_approval(
        self,
        workflow_name: str,
        step_id: str,
        payload: Any,
    ) -> bool:
        """Request approval via CLI prompt.

        Args:
            workflow_name: Name of the workflow
            step_id: ID of the step requiring approval
            payload: Data to show to approver

        Returns:
            True if approved, False if rejected
        """
        print("\n" + "=" * 60)
        print("APPROVAL REQUIRED")
        print("=" * 60)
        print(f"Workflow: {workflow_name}")
        print(f"Step: {step_id}")
        print("-" * 60)
        print("Payload:")

        if isinstance(payload, dict):
            print(json.dumps(payload, indent=2))
        else:
            print(str(payload))

        print("-" * 60)

        while True:
            response = input("Approve? [y/n]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            if response in ("n", "no"):
                return False
            print("Please enter 'y' or 'n'")


class AutoApprovalHandler(ApprovalHandler):
    """Approval handler that auto-approves everything (for testing)."""

    def __init__(self, approve: bool = True):
        """Initialize auto-approval handler.

        Args:
            approve: Whether to auto-approve (True) or auto-reject (False)
        """
        self._approve = approve

    async def request_approval(
        self,
        workflow_name: str,
        step_id: str,
        payload: Any,
    ) -> bool:
        """Auto-approve or reject.

        Args:
            workflow_name: Name of the workflow
            step_id: ID of the step requiring approval
            payload: Data to show to approver

        Returns:
            The configured approval value
        """
        return self._approve
