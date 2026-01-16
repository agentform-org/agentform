"""YAML parser for Agentform specifications."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentform_schema.models import SpecRoot


class ParseError(Exception):
    """Error during YAML parsing."""

    pass


def parse_yaml(content: str) -> SpecRoot:
    """Parse YAML content into a SpecRoot model.

    Args:
        content: YAML string content

    Returns:
        Parsed and validated SpecRoot

    Raises:
        ParseError: If YAML is invalid or doesn't match schema
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ParseError(f"Invalid YAML syntax: {e}") from e

    if not isinstance(data, dict):
        raise ParseError("YAML root must be a mapping")

    try:
        return SpecRoot.model_validate(data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            msg = error["msg"]
            errors.append(f"  - {loc}: {msg}")
        raise ParseError("Schema validation failed:\n" + "\n".join(errors)) from e


def parse_yaml_file(path: str | Path) -> SpecRoot:
    """Parse a YAML file into a SpecRoot model.

    Args:
        path: Path to YAML file

    Returns:
        Parsed and validated SpecRoot

    Raises:
        ParseError: If file not found or YAML is invalid
    """
    path = Path(path)

    if not path.exists():
        raise ParseError(f"File not found: {path}")

    try:
        content = path.read_text()
    except OSError as e:
        raise ParseError(f"Failed to read file: {e}") from e

    return parse_yaml(content)
