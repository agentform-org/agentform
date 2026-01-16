"""Init command for Agentform CLI - downloads external modules."""

from pathlib import Path

import typer
from rich.console import Console

from agentform_compiler.agentform_module_resolver import (
    ModuleResolutionError,
    ModuleResolver,
    is_git_url,
)
from agentform_compiler.agentform_parser import AgentformParseError, parse_agentform_directory

console = Console()


def init(
    directory: Path | None = typer.Argument(
        None,
        help="Path to Agentform project directory. Defaults to current directory.",
    ),
) -> None:
    """Initialize Agentform project - download external modules.

    Parses Agentform files in the directory to find module blocks with Git sources,
    then clones them to the local .agentform/modules/ directory.

    Similar to 'terraform init'.
    """
    # Resolve to absolute path
    if directory is None:
        directory = Path.cwd()
    project_dir = directory.resolve()

    if not project_dir.exists():
        console.print(f"[red]✗[/red] Directory not found: {project_dir}")
        raise typer.Exit(1)

    if not project_dir.is_dir():
        console.print(f"[red]✗[/red] Not a directory: {project_dir}")
        raise typer.Exit(1)

    # Check for .agentform files
    agentform_files = list(project_dir.glob("*.agentform"))
    if not agentform_files:
        console.print(f"[red]✗[/red] No .agentform files found in: {project_dir}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Initializing Agentform project:[/bold] {project_dir}")
    console.print(f"Found {len(agentform_files)} .agentform file(s)\n")

    # Parse the Agentform files to find module blocks
    try:
        agentform_file = parse_agentform_directory(project_dir)
    except AgentformParseError as e:
        console.print(f"[red]✗[/red] Failed to parse Agentform files:\n{e}")
        raise typer.Exit(1) from None

    # Check for modules
    if not agentform_file.modules:
        console.print("[green]✓[/green] No external modules to download")
        console.print("\n[dim]Project initialized successfully[/dim]")
        return

    # Filter to only Git modules
    git_modules = [m for m in agentform_file.modules if m.source and is_git_url(m.source)]

    if not git_modules:
        console.print("[green]✓[/green] No external Git modules to download")
        console.print("\n[dim]Project initialized successfully[/dim]")
        return

    console.print(f"[bold]Downloading {len(git_modules)} module(s)...[/bold]\n")

    # Create resolver for downloading modules
    resolver = ModuleResolver(base_path=project_dir)

    # Download each module
    success_count = 0
    error_count = 0

    for module in git_modules:
        source = module.source
        version = module.version

        # Type narrowing: source is guaranteed to be non-None after filter
        if source is None:
            continue

        console.print(f"  [dim]•[/dim] {module.name}")
        console.print(f"    Source: {source}")
        if version:
            console.print(f"    Version: {version}")

        try:
            resolved = resolver.download_module(source, version)
            console.print(f"    [green]✓[/green] Downloaded to: {resolved.path}")
            success_count += 1
        except ModuleResolutionError as e:
            console.print(f"    [red]✗[/red] Failed: {e}")
            error_count += 1

        console.print()

    # Summary
    if error_count == 0:
        console.print(f"[green]✓[/green] Successfully downloaded {success_count} module(s)")
        console.print("\n[dim]Project initialized successfully[/dim]")
    else:
        console.print(
            f"[yellow]![/yellow] Downloaded {success_count} module(s), {error_count} failed"
        )
        raise typer.Exit(1)
