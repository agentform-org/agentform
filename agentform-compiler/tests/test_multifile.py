"""Tests for multi-file Agentform support (Terraform-style file merging)."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler.agentform_ast import (
    MergeError,
    merge_agentform_files,
)
from agentform_compiler.agentform_parser import (
    AgentformParseError,
    discover_agentform_files,
    parse_agentform,
    parse_agentform_directory,
)
from agentform_compiler.compiler import (
    CompilationError,
    compile_agentform_directory,
    compile_file,
    validate_agentform_directory,
    validate_file,
)


class TestMergeAgentformFiles:
    """Test the merge_agentform_files function."""

    def test_merge_single_file(self) -> None:
        """Test merging a single file returns it unchanged."""
        content = """
        agentform { version = "0.1" project = "test" }
        variable "api_key" { type = string }
        """
        agentform_file = parse_agentform(content)
        result = merge_agentform_files([agentform_file])

        assert result.agentform is not None
        assert result.agentform.version == "0.1"
        assert len(result.variables) == 1

    def test_merge_empty_list_raises_error(self) -> None:
        """Test that merging empty list raises error."""
        with pytest.raises(MergeError, match="No Agentform files to merge"):
            merge_agentform_files([])

    def test_merge_two_files(self) -> None:
        """Test merging two files with different blocks."""
        # File 1: agentform block + variable
        file1_content = """
        agentform { version = "0.1" project = "test" }
        variable "api_key" { type = string }
        """

        # File 2: model + agent (no agentform block)
        file2_content = """
        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }
        agent "assistant" {
            model = model.gpt4
            instructions = "test"
        }
        """

        file1 = parse_agentform(file1_content, file_path="main.agentform")
        file2 = parse_agentform(file2_content, file_path="agents.agentform")

        result = merge_agentform_files([file1, file2])

        assert result.agentform is not None
        assert result.agentform.version == "0.1"
        assert len(result.variables) == 1
        assert len(result.models) == 1
        assert len(result.agents) == 1

    def test_merge_multiple_files(self) -> None:
        """Test merging multiple files with all block types."""
        # File 1: agentform + variables
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "multifile" }
            variable "openai_key" { type = string sensitive = true }
            variable "anthropic_key" { type = string sensitive = true }
            """,
            file_path="variables.agentform",
        )

        # File 2: providers + models
        file2 = parse_agentform(
            """
            provider "llm.openai" "default" { api_key = var.openai_key }
            model "gpt4" { provider = provider.llm.openai.default id = "gpt-4o" }
            model "gpt4_mini" { provider = provider.llm.openai.default id = "gpt-4o-mini" }
            """,
            file_path="providers.agentform",
        )

        # File 3: agents + policies
        file3 = parse_agentform(
            """
            policy "default" { budgets { max_cost_usd_per_run = 1.0 } }
            agent "assistant" {
                model = model.gpt4
                fallback_models = [model.gpt4_mini]
                policy = policy.default
                instructions = "Be helpful"
            }
            """,
            file_path="agents.agentform",
        )

        # File 4: workflows
        file4 = parse_agentform(
            """
            workflow "ask" {
                entry = step.process
                step "process" {
                    type = "llm"
                    agent = agent.assistant
                    next = step.end
                }
                step "end" { type = "end" }
            }
            """,
            file_path="workflows.agentform",
        )

        result = merge_agentform_files([file1, file2, file3, file4])

        assert result.agentform is not None
        assert result.agentform.project == "multifile"
        assert len(result.variables) == 2
        assert len(result.providers) == 1
        assert len(result.models) == 2
        assert len(result.policies) == 1
        assert len(result.agents) == 1
        assert len(result.workflows) == 1

    def test_merge_no_agentform_block_raises_error(self) -> None:
        """Test that merging files without agentform block raises error."""
        file1 = parse_agentform('variable "key" { type = string }', file_path="vars.agentform")
        file2 = parse_agentform(
            'model "gpt4" { provider = provider.test id = "test" }',
            file_path="models.agentform",
        )

        with pytest.raises(MergeError, match="No 'agentform' metadata block found"):
            merge_agentform_files([file1, file2])

    def test_merge_multiple_agentform_blocks_raises_error(self) -> None:
        """Test that multiple agentform blocks raises error."""
        file1 = parse_agentform(
            'agentform { version = "0.1" project = "test1" }', file_path="main.agentform"
        )
        file2 = parse_agentform(
            'agentform { version = "0.1" project = "test2" }', file_path="other.agentform"
        )

        with pytest.raises(MergeError, match="Multiple 'agentform' blocks found"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_variable_raises_error(self) -> None:
        """Test that duplicate variables raise error with file locations."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            variable "api_key" { type = string }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform('variable "api_key" { type = string }', file_path="vars.agentform")

        with pytest.raises(MergeError, match="Duplicate variable 'api_key'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_provider_raises_error(self) -> None:
        """Test that duplicate providers raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            provider "llm.openai" "default" { api_key = "test" }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'provider "llm.openai" "default" { api_key = "test2" }',
            file_path="providers.agentform",
        )

        with pytest.raises(MergeError, match=r"Duplicate provider 'llm\.openai\.default'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_model_raises_error(self) -> None:
        """Test that duplicate models raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            model "gpt4" { provider = provider.test id = "gpt-4" }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'model "gpt4" { provider = provider.test id = "gpt-4o" }',
            file_path="models.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate model 'gpt4'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_agent_raises_error(self) -> None:
        """Test that duplicate agents raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            agent "assistant" { model = model.gpt4 instructions = "v1" }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'agent "assistant" { model = model.gpt4 instructions = "v2" }',
            file_path="agents.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate agent 'assistant'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_workflow_raises_error(self) -> None:
        """Test that duplicate workflows raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            workflow "ask" {
                entry = step.end
                step "end" { type = "end" }
            }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            """
            workflow "ask" {
                entry = step.done
                step "done" { type = "end" }
            }
            """,
            file_path="workflows.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate workflow 'ask'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_server_raises_error(self) -> None:
        """Test that duplicate servers raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            server "fs" { command = ["cmd1"] transport = "stdio" }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'server "fs" { command = ["cmd2"] transport = "stdio" }',
            file_path="servers.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate server 'fs'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_capability_raises_error(self) -> None:
        """Test that duplicate capabilities raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            capability "read" { server = server.fs method = "read" }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'capability "read" { server = server.fs method = "read_v2" }',
            file_path="caps.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate capability 'read'"):
            merge_agentform_files([file1, file2])

    def test_merge_duplicate_policy_raises_error(self) -> None:
        """Test that duplicate policies raise error."""
        file1 = parse_agentform(
            """
            agentform { version = "0.1" project = "test" }
            policy "default" { budgets { max_cost_usd_per_run = 1.0 } }
            """,
            file_path="main.agentform",
        )
        file2 = parse_agentform(
            'policy "default" { budgets { max_cost_usd_per_run = 2.0 } }',
            file_path="policies.agentform",
        )

        with pytest.raises(MergeError, match="Duplicate policy 'default'"):
            merge_agentform_files([file1, file2])


class TestDiscoverAgentformFiles:
    """Test the discover_agentform_files function."""

    def test_discover_empty_directory(self) -> None:
        """Test discovering files in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = discover_agentform_files(tmpdir)
            assert files == []

    def test_discover_single_file(self) -> None:
        """Test discovering a single .agentform file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "spec.af").write_text('agentform { version = "0.1" }')
            files = discover_agentform_files(tmpdir)
            assert len(files) == 1
            assert files[0].name == "spec.af"

    def test_discover_multiple_files(self) -> None:
        """Test discovering multiple .agentform files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "agents.af").write_text('agent "test" {}')
            Path(tmpdir, "main.af").write_text('agentform { version = "0.1" }')
            Path(tmpdir, "workflows.af").write_text("workflow {}")

            files = discover_agentform_files(tmpdir)
            assert len(files) == 3
            # Should be sorted alphabetically
            assert files[0].name == "agents.af"
            assert files[1].name == "main.af"
            assert files[2].name == "workflows.af"

    def test_discover_ignores_non_agentform_files(self) -> None:
        """Test that non-.agentform files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "spec.af").write_text('agentform { version = "0.1" }')
            Path(tmpdir, "readme.md").write_text("# README")
            Path(tmpdir, "config.json").write_text("{}")
            Path(tmpdir, "script.py").write_text("print('hello')")

            files = discover_agentform_files(tmpdir)
            assert len(files) == 1
            assert files[0].name == "spec.af"

    def test_discover_case_insensitive_extension(self) -> None:
        """Test that .Af extension is also recognized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "spec.Af").write_text('agentform { version = "0.1" }')
            Path(tmpdir, "other.af").write_text("")

            files = discover_agentform_files(tmpdir)
            assert len(files) == 2

    def test_discover_nonexistent_directory_raises_error(self) -> None:
        """Test that nonexistent directory raises error."""
        with pytest.raises(AgentformParseError, match="Directory not found"):
            discover_agentform_files("/nonexistent/path")

    def test_discover_file_instead_of_directory_raises_error(self) -> None:
        """Test that passing file path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir, "spec.af")
            file_path.write_text('agentform { version = "0.1" }')

            with pytest.raises(AgentformParseError, match="not a directory"):
                discover_agentform_files(str(file_path))


class TestParseAgentformDirectory:
    """Test the parse_agentform_directory function."""

    def test_parse_empty_directory_raises_error(self) -> None:
        """Test parsing empty directory raises error."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(AgentformParseError, match=r"No \.af files found"),
        ):
            parse_agentform_directory(tmpdir)

    def test_parse_single_file_directory(self) -> None:
        """Test parsing directory with single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "spec.af").write_text(
                """
                agentform { version = "0.1" project = "single" }
                variable "key" { type = string }
                """
            )

            result = parse_agentform_directory(tmpdir)
            assert result.agentform is not None
            assert result.agentform.project == "single"
            assert len(result.variables) == 1

    def test_parse_multi_file_directory(self) -> None:
        """Test parsing directory with multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # main.agentform - contains agentform block
            Path(tmpdir, "main.af").write_text(
                """
                agentform { version = "0.2" project = "multifile-test" }
                """
            )

            # variables.agentform
            Path(tmpdir, "variables.af").write_text(
                """
                variable "openai_key" {
                    type = string
                    sensitive = true
                }
                """
            )

            # models.agentform
            Path(tmpdir, "models.af").write_text(
                """
                provider "llm.openai" "default" {
                    api_key = var.openai_key
                }
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                """
            )

            # agents.agentform
            Path(tmpdir, "agents.af").write_text(
                """
                policy "default" {
                    budgets { max_cost_usd_per_run = 0.50 }
                }
                agent "assistant" {
                    model = model.gpt4
                    policy = policy.default
                    instructions = "Be helpful"
                }
                """
            )

            # workflows.agentform
            Path(tmpdir, "workflows.af").write_text(
                """
                workflow "ask" {
                    entry = step.process
                    step "process" {
                        type = "llm"
                        agent = agent.assistant
                        next = step.end
                    }
                    step "end" { type = "end" }
                }
                """
            )

            result = parse_agentform_directory(tmpdir)

            assert result.agentform is not None
            assert result.agentform.version == "0.2"
            assert result.agentform.project == "multifile-test"
            assert len(result.variables) == 1
            assert len(result.providers) == 1
            assert len(result.models) == 1
            assert len(result.policies) == 1
            assert len(result.agents) == 1
            assert len(result.workflows) == 1

    def test_parse_directory_with_parse_error(self) -> None:
        """Test that parse errors in files are propagated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text('agentform { version = "0.1" }')
            Path(tmpdir, "broken.af").write_text("this is { invalid syntax")

            with pytest.raises(AgentformParseError):
                parse_agentform_directory(tmpdir)

    def test_parse_directory_with_merge_error(self) -> None:
        """Test that merge errors are propagated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text('agentform { version = "0.1" project = "test1" }')
            Path(tmpdir, "other.af").write_text('agentform { version = "0.1" project = "test2" }')

            with pytest.raises(MergeError, match="Multiple 'agentform' blocks"):
                parse_agentform_directory(tmpdir)


class TestCompileDirectory:
    """Test directory compilation through the compiler module."""

    def test_compile_directory_success(self) -> None:
        """Test successful directory compilation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text(
                """
                agentform { version = "0.1" project = "compiled-test" }
                variable "api_key" {
                    type = string
                    default = "test-key"
                }
                """
            )

            Path(tmpdir, "models.af").write_text(
                """
                provider "llm.openai" "default" {
                    api_key = var.api_key
                }
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                """
            )

            Path(tmpdir, "agents.af").write_text(
                """
                agent "assistant" {
                    model = model.gpt4
                    instructions = "Be helpful"
                }
                """
            )

            Path(tmpdir, "workflows.af").write_text(
                """
                workflow "ask" {
                    entry = step.end
                    step "end" { type = "end" }
                }
                """
            )

            compiled = compile_agentform_directory(tmpdir, check_env=False)

            assert compiled.project_name == "compiled-test"
            assert "openai" in compiled.providers
            assert "assistant" in compiled.agents
            assert "ask" in compiled.workflows

    def test_compile_file_detects_directory(self) -> None:
        """Test that compile_file auto-detects directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text(
                """
                agentform { version = "0.1" project = "auto-detect" }
                variable "api_key" {
                    type = string
                    default = "test-key"
                }
                agent "test" {
                    model = model.gpt4
                    instructions = "test"
                }
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                provider "llm.openai" "default" {
                    api_key = var.api_key
                }
                workflow "w" {
                    entry = step.end
                    step "end" { type = "end" }
                }
                """
            )

            compiled = compile_file(tmpdir, check_env=False)
            assert compiled.project_name == "auto-detect"

    def test_validate_directory_success(self) -> None:
        """Test successful directory validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text(
                """
                agentform { version = "0.1" project = "validate-test" }
                variable "key" {
                    type = string
                    default = "test"
                }
                """
            )

            Path(tmpdir, "agents.af").write_text(
                """
                provider "llm.openai" "default" {
                    api_key = var.key
                }
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                agent "assistant" {
                    model = model.gpt4
                    instructions = "test"
                }
                workflow "ask" {
                    entry = step.end
                    step "end" { type = "end" }
                }
                """
            )

            result = validate_agentform_directory(tmpdir, check_env=False)
            assert result.is_valid

    def test_validate_file_detects_directory(self) -> None:
        """Test that validate_file auto-detects directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text(
                """
                agentform { version = "0.1" project = "auto-validate" }
                variable "api_key" { type = string default = "test" }
                provider "llm.openai" "default" { api_key = var.api_key }
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                agent "a" { model = model.gpt4 instructions = "t" }
                workflow "w" { entry = step.end step "end" { type = "end" } }
                """
            )

            result = validate_file(tmpdir, check_env=False)
            assert result.is_valid

    def test_compile_directory_with_reference_errors(self) -> None:
        """Test that cross-file reference errors are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "main.af").write_text('agentform { version = "0.1" project = "test" }')

            # Reference undefined model
            Path(tmpdir, "agents.af").write_text(
                """
                agent "assistant" {
                    model = model.undefined_model
                    instructions = "test"
                }
                """
            )

            with pytest.raises(CompilationError, match="Unresolved reference"):
                compile_agentform_directory(tmpdir, check_env=False)

    def test_compile_directory_cross_file_references_work(self) -> None:
        """Test that references work across files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Variables in one file
            Path(tmpdir, "01_vars.af").write_text(
                """
                agentform { version = "0.1" project = "cross-ref" }
                variable "api_key" {
                    type = string
                    default = "sk-test"
                }
                """
            )

            # Provider references variable from another file
            Path(tmpdir, "02_providers.af").write_text(
                """
                provider "llm.openai" "default" {
                    api_key = var.api_key
                }
                """
            )

            # Model references provider from another file
            Path(tmpdir, "03_models.af").write_text(
                """
                model "gpt4" {
                    provider = provider.llm.openai.default
                    id = "gpt-4o"
                }
                """
            )

            # Agent references model from another file
            Path(tmpdir, "04_agents.af").write_text(
                """
                agent "assistant" {
                    model = model.gpt4
                    instructions = "Be helpful"
                }
                """
            )

            # Workflow references agent from another file
            Path(tmpdir, "05_workflows.af").write_text(
                """
                workflow "ask" {
                    entry = step.process
                    step "process" {
                        type = "llm"
                        agent = agent.assistant
                        next = step.end
                    }
                    step "end" { type = "end" }
                }
                """
            )

            compiled = compile_agentform_directory(tmpdir, check_env=False)

            assert compiled.project_name == "cross-ref"
            assert "assistant" in compiled.agents
            assert "ask" in compiled.workflows
            # Verify the agent uses the correct model
            assert compiled.agents["assistant"].model_preference == "gpt-4o"
