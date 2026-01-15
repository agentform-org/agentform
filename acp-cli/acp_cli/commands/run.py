"""Run command for ACP CLI."""

import asyncio
import json
import re
from pathlib import Path
from typing import Optional, Set

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from acp_compiler import compile_spec_file
from acp_compiler.compiler import CompilationError
from acp_runtime import WorkflowEngine
from acp_runtime.engine import WorkflowError
from acp_runtime.logging_config import configure_logging, get_logger
from acp_schema.ir import ResolvedWorkflow

console = Console()


def extract_input_fields(workflow: ResolvedWorkflow) -> Set[str]:
    """Extract all $input.field references from a workflow.
    
    Args:
        workflow: The resolved workflow to analyze
        
    Returns:
        Set of input field names (without the $input. prefix)
    """
    input_fields: Set[str] = set()
    pattern = r'\$input\.([a-zA-Z_][a-zA-Z0-9_]*)'
    
    for step in workflow.steps.values():
        # Check input_mapping (for LLM steps)
        if step.input_mapping:
            for value in step.input_mapping.values():
                if isinstance(value, str):
                    matches = re.findall(pattern, value)
                    input_fields.update(matches)
        
        # Check args_mapping (for call steps)
        if step.args_mapping:
            for value in step.args_mapping.values():
                if isinstance(value, str):
                    matches = re.findall(pattern, value)
                    input_fields.update(matches)
        
        # Check condition_expr (for condition steps)
        if step.condition_expr:
            matches = re.findall(pattern, step.condition_expr)
            input_fields.update(matches)
        
        # Check payload_expr (for human_approval steps)
        if step.payload_expr:
            matches = re.findall(pattern, step.payload_expr)
            input_fields.update(matches)
    
    return input_fields


def prompt_for_inputs(required_fields: Set[str], existing_input: dict) -> dict:
    """Prompt user for missing input fields interactively.
    
    Args:
        required_fields: Set of required input field names
        existing_input: Already provided input data
        
    Returns:
        Dictionary with all inputs (existing + prompted)
    """
    result = existing_input.copy()
    missing_fields = required_fields - set(existing_input.keys())
    
    if not missing_fields:
        return result
    
    console.print("\n[bold cyan]Missing required inputs. Please provide them:[/bold cyan]\n")
    
    for field in sorted(missing_fields):
        # Try to parse as JSON first, if that fails, use as string
        value = typer.prompt(f"  {field}")
        
        # Try to parse as JSON (for numbers, booleans, arrays, objects)
        try:
            parsed_value = json.loads(value)
            result[field] = parsed_value
        except (json.JSONDecodeError, ValueError):
            # If not valid JSON, use as string
            result[field] = value
    
    return result


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
    # Configure logging based on verbose flag
    configure_logging(verbose=verbose)
    logger = get_logger("acp_cli.run")
    
    logger.info("workflow_run_start", workflow=workflow, spec_file=str(spec_file), verbose=verbose)
    
    console.print(f"\n[bold]Running workflow:[/bold] {workflow}")
    console.print(f"[bold]Spec file:[/bold] {spec_file}\n")

    # Check spec file exists
    if not spec_file.exists():
        logger.error("spec_file_not_found", spec_file=str(spec_file))
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

    if parsed_input:
        logger.info("workflow_input", input_data=parsed_input)
        if verbose:
            console.print("[bold]Input:[/bold]")
            console.print(Syntax(json.dumps(parsed_input, indent=2), "json"))
            console.print()

    # Compile
    logger.info("compilation_start", spec_file=str(spec_file))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Compiling specification...", total=None)

        try:
            compiled = compile_spec_file(spec_file, check_env=True, resolve_credentials=True)
            logger.info(
                "compilation_success",
                workflows=list(compiled.workflows.keys()),
                agents=list(compiled.agents.keys()),
                capabilities=list(compiled.capabilities.keys()),
                servers=list(compiled.servers.keys()),
            )
        except CompilationError as e:
            logger.error("compilation_failed", error=str(e), error_type=type(e).__name__)
            console.print(f"[red]Compilation failed:[/red]\n{e}")
            raise typer.Exit(1)

    console.print("[green]✓[/green] Specification compiled")

    # Check workflow exists
    if workflow not in compiled.workflows:
        available = ", ".join(compiled.workflows.keys()) or "(none)"
        logger.error("workflow_not_found", workflow=workflow, available_workflows=list(compiled.workflows.keys()))
        console.print(f"[red]Workflow '{workflow}' not found[/red]")
        console.print(f"Available workflows: {available}")
        raise typer.Exit(1)

    # Extract required input fields and prompt if missing
    workflow_config = compiled.workflows[workflow]
    required_inputs = extract_input_fields(workflow_config)
    logger.info("workflow_config_loaded", workflow=workflow, required_inputs=list(required_inputs))
    
    if required_inputs and not parsed_input:
        # No input provided at all, prompt for all required fields
        logger.info("prompting_for_inputs", required_fields=list(required_inputs))
        parsed_input = prompt_for_inputs(required_inputs, {})
    elif required_inputs:
        # Some input provided, check for missing fields
        missing = required_inputs - set(parsed_input.keys())
        if missing:
            logger.info("prompting_for_missing_inputs", missing_fields=list(missing))
            parsed_input = prompt_for_inputs(required_inputs, parsed_input)

    # Execute
    console.print(f"\n[bold]Executing workflow...[/bold]\n")
    logger.info("workflow_execution_start", workflow=workflow, input_keys=list(parsed_input.keys()))

    engine = WorkflowEngine(compiled, verbose=verbose)

    try:
        result = asyncio.run(engine.run(workflow, parsed_input))
        logger.info("workflow_execution_success", workflow=workflow)
    except WorkflowError as e:
        logger.error("workflow_execution_failed", workflow=workflow, error=str(e), error_type=type(e).__name__)
        console.print(f"[red]Workflow execution failed:[/red]\n{e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        logger.warning("workflow_interrupted", workflow=workflow)
        console.print("\n[yellow]Workflow interrupted[/yellow]")
        raise typer.Exit(130)

    # Output results
    console.print("\n[green]✓ Workflow completed[/green]")
    
    output = result.get("output")
    state = result.get("state", {})
    logger.info("workflow_completed", workflow=workflow, has_output=output is not None, state_keys=list(state.keys()))

    if output:
        logger.debug("workflow_output", output=output)
        if output_file:
            output_file.write_text(json.dumps(output, indent=2))
            logger.info("output_written_to_file", output_file=str(output_file))
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
        logger.info("trace_written_to_file", trace_file=str(trace_file))
        console.print(f"\nTrace written to: {trace_file}")

    # Always show final state
    logger.debug("workflow_final_state", state=state)
    console.print("\n[bold]Final State:[/bold]")
    console.print(Syntax(json.dumps(state, indent=2), "json"))
    
    logger.info("workflow_run_complete", workflow=workflow)
