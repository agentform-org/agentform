"""Validate command for ACP CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from acp_compiler import parse_yaml_file
from acp_compiler.parser import ParseError
from acp_compiler.validator import validate_spec

console = Console()


def validate(
    spec_file: Path = typer.Argument(help="Path to the YAML specification file"),
    check_env: bool = typer.Option(
        True,
        "--check-env",
        help="Check if environment variables are set",
    ),
    no_check_env: bool = typer.Option(
        False,
        "--no-check-env",
        help="Skip checking environment variables",
    ),
) -> None:
    """Validate an ACP YAML specification file.

    This performs:
    - YAML syntax validation
    - Schema validation (Pydantic)
    - Reference validation (agents, capabilities, policies, etc.)
    - Environment variable checks (optional)

    Does NOT connect to MCP servers.
    """
    # Handle the two flags
    should_check_env = check_env and not no_check_env

    console.print(f"\n[bold]Validating:[/bold] {spec_file}\n")

    # Check file exists
    if not spec_file.exists():
        console.print(f"[red]✗[/red] File not found: {spec_file}")
        raise typer.Exit(1)

    # Parse
    try:
        spec = parse_yaml_file(spec_file)
        console.print("[green]✓[/green] YAML syntax valid")
        console.print("[green]✓[/green] Schema validation passed")
    except ParseError as e:
        console.print(f"[red]✗[/red] Parse error:\n{e}")
        raise typer.Exit(1)

    # Validate
    result = validate_spec(spec, check_env=should_check_env)

    # Report errors
    if result.errors:
        console.print(f"\n[red]Found {len(result.errors)} error(s):[/red]")
        for error in result.errors:
            console.print(f"  [red]✗[/red] {error.path}: {error.message}")

    # Report warnings
    if result.warnings:
        console.print(f"\n[yellow]Found {len(result.warnings)} warning(s):[/yellow]")
        for warning in result.warnings:
            console.print(f"  [yellow]![/yellow] {warning.path}: {warning.message}")

    # Summary
    if result.is_valid:
        console.print("\n[green]✓ Validation passed[/green]")

        # Print summary
        summary = []
        if spec.providers.llm:
            summary.append(f"Providers: {', '.join(spec.providers.llm.keys())}")
        if spec.servers:
            summary.append(f"Servers: {len(spec.servers)}")
        if spec.capabilities:
            summary.append(f"Capabilities: {len(spec.capabilities)}")
        if spec.agents:
            summary.append(f"Agents: {len(spec.agents)}")
        if spec.workflows:
            summary.append(f"Workflows: {len(spec.workflows)}")

        if summary:
            console.print(Panel("\n".join(summary), title="Specification Summary"))
    else:
        console.print("\n[red]✗ Validation failed[/red]")
        raise typer.Exit(1)
