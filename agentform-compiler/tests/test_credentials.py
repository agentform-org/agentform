"""Tests for credential handling."""

import os

import pytest

from agentform_compiler.credentials import (
    CredentialError,
    check_env_var_exists,
    get_env_var_name,
    is_env_reference,
    resolve_env_var,
    validate_env_references,
)


class TestIsEnvReference:
    """Tests for is_env_reference function."""

    def test_valid_env_references(self):
        """Test valid env references return True."""
        assert is_env_reference("env:OPENAI_API_KEY") is True
        assert is_env_reference("env:MY_TOKEN") is True
        assert is_env_reference("env:API_KEY_123") is True
        assert is_env_reference("env:_PRIVATE_VAR") is True
        # Case insensitive "env:" prefix
        assert is_env_reference("ENV:SOME_VAR") is True

    def test_invalid_env_references(self):
        """Test invalid env references return False."""
        assert is_env_reference("OPENAI_API_KEY") is False
        assert is_env_reference("env:") is False
        assert is_env_reference("env:123INVALID") is False  # Starts with number
        assert is_env_reference("env:has-dash") is False
        assert is_env_reference("sk-1234567890") is False
        assert is_env_reference("") is False


class TestGetEnvVarName:
    """Tests for get_env_var_name function."""

    def test_extract_valid_names(self):
        """Test extracting env var names."""
        assert get_env_var_name("env:OPENAI_API_KEY") == "OPENAI_API_KEY"
        assert get_env_var_name("env:MY_TOKEN") == "MY_TOKEN"
        assert get_env_var_name("env:A") == "A"

    def test_invalid_references_return_none(self):
        """Test invalid references return None."""
        assert get_env_var_name("OPENAI_API_KEY") is None
        assert get_env_var_name("env:") is None
        assert get_env_var_name("") is None


class TestResolveEnvVar:
    """Tests for resolve_env_var function."""

    def test_resolve_existing_var(self, monkeypatch):
        """Test resolving an existing env var."""
        monkeypatch.setenv("TEST_API_KEY", "test-value-123")

        result = resolve_env_var("env:TEST_API_KEY")
        assert result == "test-value-123"

    def test_resolve_missing_var_required(self, monkeypatch):
        """Test resolving missing required var raises error."""
        monkeypatch.delenv("MISSING_VAR", raising=False)

        with pytest.raises(CredentialError) as exc_info:
            resolve_env_var("env:MISSING_VAR", required=True)
        assert "Environment variable not set" in str(exc_info.value)

    def test_resolve_missing_var_optional(self, monkeypatch):
        """Test resolving missing optional var returns None."""
        monkeypatch.delenv("MISSING_VAR", raising=False)

        result = resolve_env_var("env:MISSING_VAR", required=False)
        assert result is None

    def test_invalid_reference_raises_error(self):
        """Test invalid reference raises CredentialError."""
        with pytest.raises(CredentialError) as exc_info:
            resolve_env_var("not-an-env-ref")
        assert "Invalid env var reference" in str(exc_info.value)


class TestCheckEnvVarExists:
    """Tests for check_env_var_exists function."""

    def test_existing_var(self, monkeypatch):
        """Test check returns True for existing var."""
        monkeypatch.setenv("EXISTS_VAR", "value")
        assert check_env_var_exists("env:EXISTS_VAR") is True

    def test_missing_var(self, monkeypatch):
        """Test check returns False for missing var."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert check_env_var_exists("env:MISSING_VAR") is False

    def test_invalid_reference(self):
        """Test check returns False for invalid reference."""
        assert check_env_var_exists("not-valid") is False


class TestValidateEnvReferences:
    """Tests for validate_env_references function."""

    def test_all_present(self, monkeypatch):
        """Test validation when all vars are present."""
        monkeypatch.setenv("VAR1", "val1")
        monkeypatch.setenv("VAR2", "val2")

        missing = validate_env_references(["env:VAR1", "env:VAR2"])
        assert missing == []

    def test_some_missing(self, monkeypatch):
        """Test validation identifies missing vars."""
        monkeypatch.setenv("VAR1", "val1")
        monkeypatch.delenv("VAR2", raising=False)
        monkeypatch.delenv("VAR3", raising=False)

        missing = validate_env_references(["env:VAR1", "env:VAR2", "env:VAR3"])
        assert set(missing) == {"VAR2", "VAR3"}

    def test_empty_list(self):
        """Test validation with empty list."""
        missing = validate_env_references([])
        assert missing == []

    def test_handles_invalid_references(self):
        """Test validation ignores invalid references."""
        missing = validate_env_references(["not-valid", "env:REAL_VAR"])
        # Invalid refs are ignored, only real vars that are missing are returned
        # REAL_VAR should be in missing if not set
        assert "REAL_VAR" in missing or os.environ.get("REAL_VAR") is not None
