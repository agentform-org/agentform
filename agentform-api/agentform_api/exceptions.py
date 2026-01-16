"""Custom exceptions for agentform-api."""


class AgentformError(Exception):
    """Base exception for Agentform API errors."""

    pass


class CompilationError(AgentformError):
    """Error during Agentform spec compilation.

    Raised when the Agentform compiler fails to parse, validate, or compile
    the specification files.
    """

    pass


class WorkflowError(AgentformError):
    """Error during workflow execution.

    Raised when a workflow fails to execute, including policy violations,
    step failures, or runtime errors.
    """

    pass
