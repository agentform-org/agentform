"""Tests for the main CLI entry point."""

from typer.testing import CliRunner

from agentform_cli.main import app

runner = CliRunner()


class TestMainCLI:
    """Tests for the main CLI application."""

    def test_cli_help(self):
        """Test that CLI shows help when no arguments provided."""
        result = runner.invoke(app, [])
        # Typer exits with code 2 when showing help for no args (no_args_is_help=True)
        assert result.exit_code == 2
        assert "Agentform - Agent as code protocol" in result.stdout

    def test_cli_version_flag_not_supported(self):
        """Test that --version is not supported (typer default)."""
        # Typer doesn't add --version by default unless explicitly configured
        result = runner.invoke(app, ["--version"])
        # Should fail or show help
        assert result.exit_code != 0

    def test_cli_commands_listed(self):
        """Test that commands are listed in help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout
        assert "run" in result.stdout

    def test_validate_command_help(self):
        """Test validate command help."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate an Agentform specification file" in result.stdout

    def test_run_command_help(self):
        """Test run command help."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an Agentform workflow" in result.stdout
