"""Logging configuration for ACP runtime."""

import logging
import sys
from typing import cast

import structlog


def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging for ACP.

    Args:
        verbose: If True, enable verbose logging (INFO level), otherwise WARNING
    """
    # Set log level
    log_level = logging.INFO if verbose else logging.WARNING

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Add renderer based on verbose mode
    if verbose:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Optional logger name (usually module name)

    Returns:
        Configured structlog logger
    """
    return cast("structlog.BoundLogger", structlog.get_logger(name))
