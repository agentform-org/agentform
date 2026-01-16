"""Validate command for Agentform CLI."""

import json
from pathlib import Path

import typer
from rich.console import Console

from agentform_compiler import CompilationError, validate_file

console = Console()


def _find_default_spec_path() -> Path:
    """Find the default spec file or directory.

    Priority:
    1. Current directory if it contains multiple .agentform files
    2. agentform.agentform file
    3. spec.agentform file
    4. Current directory (fallback)
    """
    cwd = Path()

    # Check for .agentform files in current directory
    agentform_files = list(cwd.glob("*.agentform"))

    # If multiple .agentform files exist, use directory mode
    if len(agentform_files) > 1:
        return cwd

    # Single file: use specific files
    for name in ["agentform.agentform", "spec.agentform"]:
        path = Path(name)
        if path.exists():
            return path

    # If there's exactly one .agentform file, use it
    if len(agentform_files) == 1:
        return agentform_files[0]

    # Fallback to current directory (will error later if no .agentform files)
    return cwd


def _parse_var(var_str: str) -> tuple[str, str]:
    """Parse a variable string in the form 'name=value'."""
    if "=" not in var_str:
        raise typer.BadParameter(f"Invalid variable format: {var_str}. Expected 'name=value'")
    name, value = var_str.split("=", 1)
    return name.strip(), value.strip()


def _load_variables(
    var_args: list[str] | None,
    var_file_path: Path | None,
) -> dict[str, str]:
    """Load variables from --var arguments and --var-file.

    CLI --var arguments take precedence over --var-file values.
    """
    variables: dict[str, str] = {}

    # Load from var-file first
    if var_file_path and var_file_path.exists():
        try:
            file_vars = json.loads(var_file_path.read_text())
            if isinstance(file_vars, dict):
                variables.update({k: str(v) for k, v in file_vars.items()})
            else:
                raise typer.BadParameter(
                    f"Variable file must contain a JSON object: {var_file_path}"
                )
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Error parsing variable file {var_file_path}: {e}") from None

    # Override with CLI --var arguments
    if var_args:
        for var_str in var_args:
            name, value = _parse_var(var_str)
            variables[name] = value

    return variables


def validate(
    spec_path: Path | None = typer.Argument(
        None,
        help="Path to .agentform file or directory. Defaults to current directory.",
    ),
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
    var: list[str] | None = typer.Option(
        None,
        "--var",
        help="Set a variable (can be repeated). Format: name=value",
    ),
    var_file: Path | None = typer.Option(
        None,
        "--var-file",
        help="Path to JSON file with variables",
    ),
) -> None:
    """Validate an Agentform specification.

    Runs from the current directory by default, automatically discovering and
    merging all .agentform files (Terraform-style).

    This performs:
    - Syntax validation
    - Schema validation (Pydantic)
    - Reference validation (agents, capabilities, policies, etc.)
    - Variable validation (optional)

    Does NOT connect to MCP servers.
    """
    # Handle the two flags
    should_check_env = check_env and not no_check_env

    # Auto-detect spec path if not provided
    if spec_path is None:
        spec_path = _find_default_spec_path()

    # Determine if input is file or directory
    is_directory = spec_path.is_dir()

    if is_directory:
        agentform_files = list(spec_path.glob("*.agentform"))
        console.print(
            f"\n[bold]Validating {len(agentform_files)} .agentform file(s) from:[/bold] {spec_path.resolve()}\n"
        )
    else:
        console.print(f"\n[bold]Validating:[/bold] {spec_path}\n")

    # Check path exists
    if not spec_path.exists():
        console.print(f"[red]✗[/red] Path not found: {spec_path}")
        raise typer.Exit(1)

    # Load variables
    try:
        variables = _load_variables(var, var_file)
    except typer.BadParameter as e:
        console.print(f"[red]Error loading variables:[/red] {e}")
        raise typer.Exit(1) from None

    # Validate
    try:
        result = validate_file(spec_path, check_env=should_check_env, variables=variables)
        if is_directory:
            console.print("[green]✓[/green] Agentform syntax valid (merged from directory)")
        else:
            console.print("[green]✓[/green] Agentform syntax valid")
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

            from agentform_compiler import compile_file

            compiled = compile_file(
                spec_path,
                check_env=False,
                resolve_credentials=False,
                variables=variables,
            )

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
