"""Run command for ACP CLI."""

import asyncio
import json
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

from acp_compiler import compile_file
from acp_compiler.compiler import CompilationError
from acp_runtime import WorkflowEngine
from acp_runtime.engine import WorkflowError
from acp_runtime.logging_config import configure_logging, get_logger
from acp_schema.ir import ResolvedWorkflow

console = Console()


def extract_input_fields(workflow: ResolvedWorkflow) -> set[str]:
    """Extract all $input.field references from a workflow.

    Args:
        workflow: The resolved workflow to analyze

    Returns:
        Set of input field names (without the $input. prefix)
    """
    input_fields: set[str] = set()
    pattern = r"\$input\.([a-zA-Z_][a-zA-Z0-9_]*)"

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


def prompt_for_inputs(required_fields: set[str], existing_input: dict) -> dict:
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


def _find_default_spec_path() -> Path:
    """Find the default spec file or directory.

    Priority:
    1. Current directory if it contains multiple .acp files
    2. acp.acp file
    3. spec.acp file
    4. Current directory (fallback)
    """
    cwd = Path(".")

    # Check for .acp files in current directory
    acp_files = list(cwd.glob("*.acp"))

    # If multiple .acp files exist, use directory mode
    if len(acp_files) > 1:
        return cwd

    # Single file: use specific files
    for name in ["acp.acp", "spec.acp"]:
        path = Path(name)
        if path.exists():
            return path

    # If there's exactly one .acp file, use it
    if len(acp_files) == 1:
        return acp_files[0]

    # Fallback to current directory (will error later if no .acp files)
    return cwd


def _parse_var(var_str: str) -> tuple[str, str]:
    """Parse a variable string in the form 'name=value'."""
    if "=" not in var_str:
        raise typer.BadParameter(f"Invalid variable format: {var_str}. Expected 'name=value'")
    name, value = var_str.split("=", 1)
    return name.strip(), value.strip()


def _load_variables(
    var_args: list[str] | None,
    var_file: Path | None,
) -> dict[str, str]:
    """Load variables from --var arguments and --var-file.

    CLI --var arguments take precedence over --var-file values.
    """
    variables: dict[str, str] = {}

    # Load from var-file first
    if var_file and var_file.exists():
        try:
            file_vars = json.loads(var_file.read_text())
            if isinstance(file_vars, dict):
                variables.update({k: str(v) for k, v in file_vars.items()})
            else:
                raise typer.BadParameter(f"Variable file must contain a JSON object: {var_file}")
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f"Error parsing variable file {var_file}: {e}") from None

    # Override with CLI --var arguments
    if var_args:
        for var_str in var_args:
            name, value = _parse_var(var_str)
            variables[name] = value

    return variables


def run(
    workflow: str = typer.Argument(help="Name of the workflow to run"),
    path: Path | None = typer.Argument(
        None,
        help="Path to .acp file or directory. Defaults to current directory.",
    ),
    input_data: str | None = typer.Option(
        None,
        "--input",
        "-i",
        help="JSON input data for the workflow (string or @file.json)",
    ),
    input_file: Path | None = typer.Option(
        None,
        "--input-file",
        "-f",
        help="Path to JSON file with input data",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write output to file instead of stdout",
    ),
    trace_file: Path | None = typer.Option(
        None,
        "--trace",
        "-t",
        help="Write execution trace to file",
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
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """Run an ACP workflow.

    Runs from the current directory by default, automatically discovering and
    merging all .acp files (Terraform-style). Optionally specify a path.

    This will:
    1. Discover and compile all .acp files
    2. Connect to MCP servers (if any)
    3. Execute the specified workflow
    4. Output the result
    """
    # Configure logging based on verbose flag
    configure_logging(verbose=verbose)
    logger = get_logger("acp_cli.run")

    # Auto-detect spec path if not provided
    if path is None:
        spec_path = _find_default_spec_path()
    else:
        spec_path = path

    # Check spec path exists
    if not spec_path.exists():
        path_type = "directory" if spec_path.suffix == "" else "file"
        logger.error("spec_path_not_found", spec_path=str(spec_path))
        console.print(f"[red]Spec {path_type} not found:[/red] {spec_path}")
        raise typer.Exit(1)

    # Determine if input is file or directory
    is_directory = spec_path.is_dir()

    logger.info("workflow_run_start", workflow=workflow, spec_path=str(spec_path), verbose=verbose)

    console.print(f"\n[bold]Running workflow:[/bold] {workflow}")
    if is_directory:
        acp_files = list(spec_path.glob("*.acp"))
        console.print(f"[bold]Using {len(acp_files)} .acp file(s) from:[/bold] {spec_path.resolve()}\n")
    else:
        console.print(f"[bold]Using spec:[/bold] {spec_path}\n")

    # Load variables
    try:
        variables = _load_variables(var, var_file)
    except typer.BadParameter as e:
        console.print(f"[red]Error loading variables:[/red] {e}")
        raise typer.Exit(1) from None

    if variables and verbose:
        # Mask sensitive values in output
        display_vars = {
            k: "***"
            if "key" in k.lower() or "secret" in k.lower() or "password" in k.lower()
            else v
            for k, v in variables.items()
        }
        console.print(f"[bold]Variables:[/bold] {display_vars}\n")

    # Parse input data
    parsed_input: dict = {}

    if input_file and input_file.exists():
        try:
            parsed_input = json.loads(input_file.read_text())
        except json.JSONDecodeError as e:
            console.print(f"[red]Error parsing input file:[/red] {e}")
            raise typer.Exit(1) from None
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
                raise typer.Exit(1) from None
        else:
            try:
                parsed_input = json.loads(input_data)
            except json.JSONDecodeError as e:
                console.print(f"[red]Error parsing input JSON:[/red] {e}")
                raise typer.Exit(1) from None

    if parsed_input:
        logger.info("workflow_input", input_data=parsed_input)
        if verbose:
            console.print("[bold]Input:[/bold]")
            console.print(Syntax(json.dumps(parsed_input, indent=2), "json"))
            console.print()

    # Compile
    logger.info("compilation_start", spec_path=str(spec_path))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task_desc = "Compiling specification..."
        if is_directory:
            task_desc = "Compiling specification (merging files)..."
        progress.add_task(task_desc, total=None)

        try:
            compiled = compile_file(
                spec_path,
                check_env=True,
                resolve_credentials=True,
                variables=variables,
            )
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
            raise typer.Exit(1) from None

    console.print("[green]✓[/green] Specification compiled")

    # Check workflow exists
    if workflow not in compiled.workflows:
        available = ", ".join(compiled.workflows.keys()) or "(none)"
        logger.error(
            "workflow_not_found",
            workflow=workflow,
            available_workflows=list(compiled.workflows.keys()),
        )
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
    console.print("\n[bold]Executing workflow...[/bold]\n")
    logger.info("workflow_execution_start", workflow=workflow, input_keys=list(parsed_input.keys()))

    engine = WorkflowEngine(compiled, verbose=verbose)

    try:
        result = asyncio.run(engine.run(workflow, parsed_input))
        logger.info("workflow_execution_success", workflow=workflow)
    except WorkflowError as e:
        logger.error(
            "workflow_execution_failed",
            workflow=workflow,
            error=str(e),
            error_type=type(e).__name__,
        )
        console.print(f"[red]Workflow execution failed:[/red]\n{e}")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        logger.warning("workflow_interrupted", workflow=workflow)
        console.print("\n[yellow]Workflow interrupted[/yellow]")
        raise typer.Exit(130) from None

    # Output results
    console.print("\n[green]✓ Workflow completed[/green]")

    output = result.get("output")
    state = result.get("state", {})
    logger.info(
        "workflow_completed",
        workflow=workflow,
        has_output=output is not None,
        state_keys=list(state.keys()),
    )

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
