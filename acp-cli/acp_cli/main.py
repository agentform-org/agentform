"""Main CLI entry point for ACP."""

import typer
from rich.console import Console

from acp_cli.commands.compile import compile_cmd
from acp_cli.commands.init import init
from acp_cli.commands.run import run
from acp_cli.commands.validate import validate

console = Console()

app = typer.Typer(
    name="acp",
    help="ACP - Agent as code protocol",
    add_completion=False,
    no_args_is_help=True,
)

# Register commands
app.command(name="init", help="Initialize project - download external modules")(init)
app.command(name="validate", help="Validate an ACP specification file")(validate)
app.command(name="compile", help="Compile an ACP specification to IR")(compile_cmd)
app.command(name="run", help="Run an ACP workflow")(run)


@app.callback()
def callback() -> None:
    """ACP - Agent as code protocol.

    A declarative framework for defining AI agent systems using YAML.
    """
    pass


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
