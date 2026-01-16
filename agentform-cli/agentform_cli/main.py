"""Main CLI entry point for Agentform."""

import typer
from rich.console import Console

from agentform_cli.commands.compile import compile_cmd
from agentform_cli.commands.init import init
from agentform_cli.commands.run import run
from agentform_cli.commands.validate import validate

console = Console()

app = typer.Typer(
    name="agentform",
    help="Agentform - Agent as code protocol",
    add_completion=False,
    no_args_is_help=True,
)

# Register commands
app.command(name="init", help="Initialize project - download external modules")(init)
app.command(name="validate", help="Validate an Agentform specification file")(validate)
app.command(name="compile", help="Compile an Agentform specification to IR")(compile_cmd)
app.command(name="run", help="Run an Agentform workflow")(run)


@app.callback()
def callback() -> None:
    """Agentform - Agent as code protocol.

    A declarative framework for defining AI agent systems using YAML.
    """
    pass


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
