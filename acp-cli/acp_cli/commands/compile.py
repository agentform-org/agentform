"""Compile command for ACP CLI."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax

from acp_compiler import compile_file
from acp_compiler.compiler import CompilationError

console = Console()

# Supported file extensions
ACP_EXTENSIONS = {".acp"}
YAML_EXTENSIONS = {".yaml", ".yml"}


def _get_file_type(path: Path) -> str:
    """Get file type description for display."""
    if path.suffix.lower() in ACP_EXTENSIONS:
        return "ACP"
    return "YAML"


def compile_cmd(
    spec_file: Path = typer.Argument(help="Path to the specification file (.acp, .yaml, or .yml)"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write compiled IR to file (JSON format)",
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--compact",
        help="Pretty-print JSON output",
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
    resolve_credentials: bool = typer.Option(
        False,
        "--resolve-credentials",
        help="Resolve environment variables to actual values (security risk!)",
    ),
) -> None:
    """Compile an ACP specification to IR (Intermediate Representation).

    Supports both .acp (native schema) and .yaml/.yml (YAML) formats.
    Auto-detects format based on file extension.

    Outputs the compiled IR as JSON, useful for debugging and tooling.
    """
    # Handle the two flags
    should_check_env = check_env and not no_check_env

    file_type = _get_file_type(spec_file)
    console.print(f"\n[bold]Compiling ({file_type}):[/bold] {spec_file}\n")

    # Check file exists
    if not spec_file.exists():
        console.print(f"[red]✗[/red] File not found: {spec_file}")
        raise typer.Exit(1)

    # Compile
    try:
        compiled = compile_file(
            spec_file,
            check_env=should_check_env,
            resolve_credentials=resolve_credentials,
        )
        console.print("[green]✓[/green] Compilation successful")
    except CompilationError as e:
        console.print(f"[red]✗[/red] Compilation failed:\n{e}")
        raise typer.Exit(1) from None

    # Convert to JSON
    ir_dict = compiled.model_dump(mode="json")

    # Remove resolved credential values for security (unless explicitly requested)
    if not resolve_credentials:
        _strip_credential_values(ir_dict)

    indent = 2 if pretty else None
    ir_json = json.dumps(ir_dict, indent=indent)

    # Output
    if output:
        output.write_text(ir_json)
        console.print(f"\n[green]✓[/green] IR written to: {output}")
    else:
        console.print("\n[bold]Compiled IR:[/bold]")
        console.print(Syntax(ir_json, "json"))

    # Print summary
    summary = []
    if compiled.providers:
        summary.append(f"Providers: {len(compiled.providers)}")
    if compiled.servers:
        summary.append(f"Servers: {len(compiled.servers)}")
    if compiled.capabilities:
        summary.append(f"Capabilities: {len(compiled.capabilities)}")
    if compiled.policies:
        summary.append(f"Policies: {len(compiled.policies)}")
    if compiled.agents:
        summary.append(f"Agents: {len(compiled.agents)}")
    if compiled.workflows:
        summary.append(f"Workflows: {len(compiled.workflows)}")

    if summary:
        console.print(f"\n[dim]{', '.join(summary)}[/dim]")


def _strip_credential_values(ir_dict: dict) -> None:
    """Remove credential values from IR dict for security."""
    # Strip from providers
    for provider in ir_dict.get("providers", {}).values():
        if "api_key" in provider and isinstance(provider["api_key"], dict):
            provider["api_key"]["value"] = None

    # Strip from servers
    for server in ir_dict.get("servers", {}).values():
        if "auth_token" in server and isinstance(server["auth_token"], dict):
            server["auth_token"]["value"] = None

