"""Integration tests for Agentform module system."""

import tempfile
from pathlib import Path

import pytest

from agentform_compiler.agentform_parser import parse_agentform
from agentform_compiler.agentform_resolver import add_module_symbols, resolve_references
from agentform_compiler.compiler import compile_agentform_directory


class TestModuleParsing:
    """Tests for parsing module blocks."""

    def test_parses_module_block(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        module "my-module" {
            source  = "github.com/example/module"
            version = "v1.0.0"
            api_key = "secret-key"
        }
        """
        result = parse_agentform(code)

        assert len(result.modules) == 1
        module = result.modules[0]
        assert module.name == "my-module"
        assert module.source == "github.com/example/module"
        assert module.version == "v1.0.0"
        params = module.get_parameters()
        assert params["api_key"] == "secret-key"

    def test_parses_multiple_modules(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        module "module-a" {
            source = "./modules/a"
        }

        module "module-b" {
            source = "./modules/b"
            param1 = "value1"
        }
        """
        result = parse_agentform(code)

        assert len(result.modules) == 2
        assert result.modules[0].name == "module-a"
        assert result.modules[1].name == "module-b"

    def test_parses_module_with_var_ref_param(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        variable "api_key" {
            type = string
        }

        module "llm" {
            source  = "./modules/llm"
            api_key = var.api_key
        }
        """
        result = parse_agentform(code)

        assert len(result.modules) == 1
        module = result.modules[0]
        params = module.get_parameters()
        # Should be a VarRef, not a resolved string
        from agentform_compiler.agentform_ast import VarRef

        assert isinstance(params["api_key"], VarRef)
        assert params["api_key"].var_name == "api_key"


class TestModuleSymbols:
    """Tests for module symbol resolution."""

    def test_builds_module_symbols(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        module "my-module" {
            source = "./modules/test"
        }
        """
        agentform_file = parse_agentform(code)
        resolution = resolve_references(agentform_file)

        # Module itself should be registered
        assert "module.my-module" in resolution.symbols
        assert resolution.symbols["module.my-module"].kind == "module"

    def test_validates_module_reference(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        module "existing" {
            source = "./modules/test"
        }

        agent "test" {
            model = module.existing.model.default
            policy = policy.default
        }

        policy "default" {
            budgets { timeout_seconds = 60 }
        }
        """
        agentform_file = parse_agentform(code)
        resolution = resolve_references(agentform_file)

        # Should have the module registered
        assert "module.existing" in resolution.symbols
        # No errors about the module reference (it's validated at load time)
        assert resolution.is_valid

    def test_catches_unresolved_module_reference(self) -> None:
        code = """
        agentform { version = "0.1" project = "test" }

        agent "test" {
            model = module.nonexistent.model.default
            policy = policy.default
        }

        policy "default" {
            budgets { timeout_seconds = 60 }
        }
        """
        agentform_file = parse_agentform(code)
        resolution = resolve_references(agentform_file)

        # Should have an error about unresolved module
        assert not resolution.is_valid
        assert any("nonexistent" in str(e) for e in resolution.errors)


class TestAddModuleSymbols:
    """Tests for add_module_symbols function."""

    def test_adds_namespaced_symbols(self) -> None:
        # Parse a module's content
        module_code = """
        agentform { version = "0.1" project = "test-module" }

        provider "llm.openai" "default" {
            api_key = "test"
        }

        model "gpt4" {
            provider = provider.llm.openai.default
            id = "gpt-4o"
        }

        agent "assistant" {
            model = model.gpt4
            policy = policy.default
        }

        policy "default" {
            budgets { timeout_seconds = 60 }
        }
        """
        module_agentform = parse_agentform(module_code)

        # Create a main file resolution
        main_code = """
        agentform { version = "0.1" project = "main" }

        module "llm" {
            source = "./modules/llm"
        }
        """
        main_agentform = parse_agentform(main_code)
        resolution = resolve_references(main_agentform)

        # Add module symbols
        add_module_symbols(resolution, "llm", module_agentform)

        # Check that module resources are namespaced
        assert "module.llm.provider.llm.openai.default" in resolution.symbols
        assert "module.llm.model.gpt4" in resolution.symbols
        assert "module.llm.agent.assistant" in resolution.symbols
        assert "module.llm.policy.default" in resolution.symbols


class TestModuleIntegration:
    """Integration tests for full module compilation."""

    @pytest.fixture
    def test_project_dir(self) -> Path:
        """Create a temporary project with a module reference."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "modules"
        if not (fixtures_dir / "simple-module").exists():
            pytest.skip("Fixture module not found")
        return fixtures_dir

    def test_compiles_project_with_local_module(self, test_project_dir: Path) -> None:
        """Test compiling a project that uses a local module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)

            # Create main project file
            main_file = project_dir / "main.af"
            main_file.write_text(f"""
            agentform {{
                version = "0.1"
                project = "test-with-module"
            }}

            module "llm" {{
                source  = "{test_project_dir / "simple-module"}"
                api_key = "test-key-123"
            }}

            // Reference module resources
            agent "my_agent" {{
                model = module.llm.model.default
                policy = module.llm.policy.standard
                instructions = "Hello"
            }}

            workflow "main" {{
                entry = step.ask

                step "ask" {{
                    type = "llm"
                    agent = agent.my_agent
                    input {{
                        question = input.query
                    }}
                    next = step.end
                }}

                step "end" {{
                    type = "end"
                }}
            }}
            """)

            # Compile the project
            result = compile_agentform_directory(
                project_dir,
                check_env=False,  # Don't check env vars
                resolve_credentials=False,  # Don't resolve credentials
            )

            # Verify module resources are included
            assert result.agents is not None
            assert "my_agent" in result.agents

            # Module resources should be namespaced
            assert result.policies is not None
            assert "module.llm.standard" in result.policies
