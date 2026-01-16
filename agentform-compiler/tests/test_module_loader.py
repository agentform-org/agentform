"""Tests for Agentform module loader."""

from pathlib import Path

import pytest

from agentform_compiler.agentform_ast import Attribute, ModuleBlock
from agentform_compiler.agentform_module_loader import (
    ModuleLoader,
    ModuleLoadError,
)


class TestModuleLoader:
    """Tests for ModuleLoader."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "modules"

    @pytest.fixture
    def simple_module_path(self, fixtures_dir: Path) -> Path:
        path = fixtures_dir / "simple-module"
        if not path.exists():
            pytest.skip("Fixture module not found")
        return path

    def test_loads_module_with_required_params(
        self, simple_module_path: Path, fixtures_dir: Path
    ) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                Attribute(name="api_key", value="test-api-key-123"),
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)
        loaded = loader.load_module(module_block)

        assert loaded.name == "test-module"
        assert loaded.path == simple_module_path
        assert loaded.parameters["api_key"] == "test-api-key-123"
        assert loaded.parameters["model_name"] == "gpt-4o-mini"  # default
        assert loaded.parameters["temperature"] == 0.7  # default

    def test_raises_for_missing_source(self, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="api_key", value="test-key"),
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)

        with pytest.raises(ModuleLoadError) as exc_info:
            loader.load_module(module_block)

        assert "missing required 'source'" in str(exc_info.value)

    def test_raises_for_missing_required_param(self, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                # Missing required 'api_key'
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)

        with pytest.raises(ModuleLoadError) as exc_info:
            loader.load_module(module_block)

        assert "requires parameter 'api_key'" in str(exc_info.value)

    def test_uses_default_values(self, simple_module_path: Path, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                Attribute(name="api_key", value="test-key"),
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)
        loaded = loader.load_module(module_block)

        # Should use defaults for model_name and temperature
        assert loaded.parameters["model_name"] == "gpt-4o-mini"
        assert loaded.parameters["temperature"] == 0.7

    def test_overrides_default_values(self, simple_module_path: Path, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                Attribute(name="api_key", value="test-key"),
                Attribute(name="model_name", value="gpt-4o"),
                Attribute(name="temperature", value=0.3),
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)
        loaded = loader.load_module(module_block)

        assert loaded.parameters["model_name"] == "gpt-4o"
        assert loaded.parameters["temperature"] == 0.3

    def test_validates_param_type(self, simple_module_path: Path, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                Attribute(name="api_key", value="test-key"),
                Attribute(name="temperature", value="not-a-number"),  # Wrong type
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)

        with pytest.raises(ModuleLoadError) as exc_info:
            loader.load_module(module_block)

        assert "expects number" in str(exc_info.value)


class TestLoadedModule:
    """Tests for LoadedModule."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "modules"

    def test_get_exported_resources(self, fixtures_dir: Path) -> None:
        module_block = ModuleBlock(
            name="test-module",
            attributes=[
                Attribute(name="source", value="simple-module"),
                Attribute(name="api_key", value="test-key"),
            ],
        )

        loader = ModuleLoader(base_path=fixtures_dir)
        loaded = loader.load_module(module_block)

        resources = loaded.get_exported_resources()

        assert "llm.openai.default" in resources["providers"]
        assert "standard" in resources["policies"]
        assert "default" in resources["models"]
        assert "assistant" in resources["agents"]


class TestLoadMultipleModules:
    """Tests for loading multiple modules."""

    @pytest.fixture
    def fixtures_dir(self) -> Path:
        return Path(__file__).parent / "fixtures" / "modules"

    def test_loads_multiple_modules(self, fixtures_dir: Path) -> None:
        module_blocks = [
            ModuleBlock(
                name="module-a",
                attributes=[
                    Attribute(name="source", value="simple-module"),
                    Attribute(name="api_key", value="key-a"),
                ],
            ),
            ModuleBlock(
                name="module-b",
                attributes=[
                    Attribute(name="source", value="simple-module"),
                    Attribute(name="api_key", value="key-b"),
                ],
            ),
        ]

        loader = ModuleLoader(base_path=fixtures_dir)
        loaded = loader.load_modules(module_blocks)

        assert "module-a" in loaded
        assert "module-b" in loaded
        assert loaded["module-a"].parameters["api_key"] == "key-a"
        assert loaded["module-b"].parameters["api_key"] == "key-b"

    def test_raises_for_duplicate_module_names(self, fixtures_dir: Path) -> None:
        module_blocks = [
            ModuleBlock(
                name="same-name",
                attributes=[
                    Attribute(name="source", value="simple-module"),
                    Attribute(name="api_key", value="key-1"),
                ],
            ),
            ModuleBlock(
                name="same-name",
                attributes=[
                    Attribute(name="source", value="simple-module"),
                    Attribute(name="api_key", value="key-2"),
                ],
            ),
        ]

        loader = ModuleLoader(base_path=fixtures_dir)

        with pytest.raises(ModuleLoadError) as exc_info:
            loader.load_modules(module_blocks)

        assert "Duplicate module name" in str(exc_info.value)
