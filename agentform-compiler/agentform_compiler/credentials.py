"""Environment variable resolution for credentials."""

import os
import re


class CredentialError(Exception):
    """Error resolving credentials."""

    pass


ENV_VAR_PATTERN = re.compile(r"^env:([A-Z_][A-Z0-9_]*)$", re.IGNORECASE)


def is_env_reference(value: str) -> bool:
    """Check if a value is an environment variable reference.

    Args:
        value: String to check

    Returns:
        True if value is in format "env:VAR_NAME"
    """
    return bool(ENV_VAR_PATTERN.match(value))


def get_env_var_name(value: str) -> str | None:
    """Extract environment variable name from reference.

    Args:
        value: String in format "env:VAR_NAME"

    Returns:
        Variable name or None if not a valid reference
    """
    match = ENV_VAR_PATTERN.match(value)
    if match:
        return match.group(1)
    return None


def resolve_env_var(value: str, required: bool = True) -> str | None:
    """Resolve an environment variable reference.

    Args:
        value: String in format "env:VAR_NAME"
        required: If True, raise error when var is not set

    Returns:
        Resolved value or None if not set and not required

    Raises:
        CredentialError: If reference is invalid or var not set
    """
    var_name = get_env_var_name(value)
    if var_name is None:
        raise CredentialError(f"Invalid env var reference: {value}")

    env_value = os.environ.get(var_name)

    if env_value is None and required:
        raise CredentialError(f"Environment variable not set: {var_name}")

    return env_value


def check_env_var_exists(value: str) -> bool:
    """Check if an environment variable exists.

    Args:
        value: String in format "env:VAR_NAME"

    Returns:
        True if variable is set
    """
    var_name = get_env_var_name(value)
    if var_name is None:
        return False
    return var_name in os.environ


def validate_env_references(references: list[str]) -> list[str]:
    """Validate that all env references are set.

    Args:
        references: List of env:VAR_NAME references

    Returns:
        List of missing variable names
    """
    missing = []
    for ref in references:
        var_name = get_env_var_name(ref)
        if var_name and var_name not in os.environ:
            missing.append(var_name)
    return missing
