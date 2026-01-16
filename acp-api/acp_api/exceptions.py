"""Custom exceptions for acp-api."""


class ACPError(Exception):
    """Base exception for ACP API errors."""

    pass


class CompilationError(ACPError):
    """Error during ACP spec compilation.

    Raised when the ACP compiler fails to parse, validate, or compile
    the specification files.
    """

    pass


class WorkflowError(ACPError):
    """Error during workflow execution.

    Raised when a workflow fails to execute, including policy violations,
    step failures, or runtime errors.
    """

    pass
