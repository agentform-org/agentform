"""Validate command for ACP CLI."""

from pathlib import Path

import typer
from rich.console import Console

from acp_compiler import CompilationError, validate_file

console = Console()


def validate(
    spec_file: Path = typer.Argument(help="Path to the .acp specification file"),
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
    """Validate an ACP specification file.

    This performs:
    - Syntax validation
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

    # Validate
    try:
        result = validate_file(spec_file, check_env=should_check_env)
        console.print("[green]✓[/green] ACP syntax valid")
        console.print("[green]✓[/green] Schema validation passed")
    except CompilationError as e:
        console.print(f"[red]✗[/red] Parse error:\n{e}")
        raise typer.Exit(1) from None

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

        # Print summary by compiling and inspecting the spec
        try:
            from rich.panel import Panel

            from acp_compiler import compile_file

            compiled = compile_file(spec_file, check_env=False, resolve_credentials=False)

            summary = []
            if compiled.providers:
                summary.append(f"Providers: {', '.join(compiled.providers.keys())}")
            if compiled.servers:
                summary.append(f"Servers: {len(compiled.servers)}")
            if compiled.capabilities:
                summary.append(f"Capabilities: {len(compiled.capabilities)}")
            if compiled.agents:
                summary.append(f"Agents: {len(compiled.agents)}")
            if compiled.workflows:
                summary.append(f"Workflows: {len(compiled.workflows)}")

            if summary:
                console.print(Panel("\n".join(summary), title="Specification Summary"))
        except Exception:
            # If compilation fails for any reason, just skip the summary
            pass
    else:
        console.print("\n[red]✗ Validation failed[/red]")
        raise typer.Exit(1)
