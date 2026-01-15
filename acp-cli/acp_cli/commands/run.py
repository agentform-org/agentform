"""Run command for ACP CLI."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from acp_compiler import compile_spec_file
from acp_compiler.compiler import CompilationError
from acp_runtime import WorkflowEngine
from acp_runtime.engine import WorkflowError

console = Console()


def run(
    workflow: str = typer.Argument(help="Name of the workflow to run"),
    spec_file: Path = typer.Option(
        Path("acp.yaml"),
        "--spec", "-s",
        help="Path to the YAML specification file",
    ),
    input_data: Optional[str] = typer.Option(
        None,
        "--input", "-i",
        help="JSON input data for the workflow (string or @file.json)",
    ),
    input_file: Optional[Path] = typer.Option(
        None,
        "--input-file", "-f",
        help="Path to JSON file with input data",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Write output to file instead of stdout",
    ),
    trace_file: Optional[Path] = typer.Option(
        None,
        "--trace", "-t",
        help="Write execution trace to file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show verbose output",
    ),
) -> None:
    """Run an ACP workflow.

    This will:
    1. Compile the YAML specification
    2. Connect to MCP servers (if any)
    3. Execute the specified workflow
    4. Output the result
    """
    console.print(f"\n[bold]Running workflow:[/bold] {workflow}")
    console.print(f"[bold]Spec file:[/bold] {spec_file}\n")

    # Check spec file exists
    if not spec_file.exists():
        console.print(f"[red]Spec file not found:[/red] {spec_file}")
        raise typer.Exit(1)

    # Parse input data
    parsed_input: dict = {}

    if input_file and input_file.exists():
        try:
            parsed_input = json.loads(input_file.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing input file:[/red] {e}")
            raise typer.Exit(1)
    elif input_data:
        if input_data.startswith("@"):
            # Load from file
            file_path = Path(input_data[1:])
            if not file_path.exists():
                console.print(f"[red]Input file not found:[/red] {file_path}")
                raise typer.Exit(1)
            try:
                parsed_input = json.loads(file_path.read_text())
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing input file:[/red] {e}")
                raise typer.Exit(1)
        else:
            try:
                parsed_input = json.loads(input_data)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing input JSON:[/red] {e}")
                raise typer.Exit(1)

    if verbose and parsed_input:
        console.print("[bold]Input:[/bold]")
        console.print(Syntax(json.dumps(parsed_input, indent=2), "json"))
        console.print()

    # Compile
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Compiling specification...", total=None)

        try:
            compiled = compile_spec_file(spec_file, check_env=True, resolve_credentials=True)
        except CompilationError as e:
            console.print(f"[red]Compilation failed:[/red]\n{e}")
            raise typer.Exit(1)

    console.print("[green]✓[/green] Specification compiled")

    # Check workflow exists
    if workflow not in compiled.workflows:
        available = ", ".join(compiled.workflows.keys()) or "(none)"
        console.print(f"[red]Workflow '{workflow}' not found[/red]")
        console.print(f"Available workflows: {available}")
        raise typer.Exit(1)

    # Execute
    console.print(f"\n[bold]Executing workflow...[/bold]\n")

    engine = WorkflowEngine(compiled)

    try:
        result = asyncio.run(engine.run(workflow, parsed_input))
    except WorkflowError as e:
        console.print(f"[red]Workflow execution failed:[/red]\n{e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Workflow interrupted[/yellow]")
        raise typer.Exit(130)

    # Output results
    console.print("\n[green]✓ Workflow completed[/green]")

    output = result.get("output")
    if output:
        if output_file:
            output_file.write_text(json.dumps(output, indent=2))
            console.print(f"\nOutput written to: {output_file}")
        else:
            console.print("\n[bold]Output:[/bold]")
            if isinstance(output, dict):
                console.print(Syntax(json.dumps(output, indent=2), "json"))
            else:
                console.print(str(output))

    # Write trace if requested
    if trace_file:
        trace = result.get("trace", "{}")
        trace_file.write_text(trace)
        console.print(f"\nTrace written to: {trace_file}")

    if verbose:
        state = result.get("state", {})
        console.print("\n[bold]Final State:[/bold]")
        console.print(Syntax(json.dumps(state, indent=2), "json"))
