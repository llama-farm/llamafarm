"""Tests for EnvService."""

import os
import tempfile
from pathlib import Path

from services.env_service import EnvService


class TestEnvService:
    """Tests for EnvService."""

    def setup_method(self):
        """Clear cache before each test."""
        EnvService.clear_cache()

    # ==========================================================================
    # Basic ${VAR} substitution
    # ==========================================================================

    def test_substitute_simple_var(self):
        """Test simple ${VAR} substitution from os.environ."""
        original = os.environ.get("TEST_VAR_SIMPLE")
        try:
            os.environ["TEST_VAR_SIMPLE"] = "hello_world"
            result = EnvService.substitute_env_vars("${TEST_VAR_SIMPLE}")
            assert result == "hello_world"
        finally:
            if original:
                os.environ["TEST_VAR_SIMPLE"] = original
            else:
                os.environ.pop("TEST_VAR_SIMPLE", None)

    def test_substitute_var_with_surrounding_text(self):
        """Test ${VAR} substitution with surrounding text."""
        original = os.environ.get("TEST_API_KEY")
        try:
            os.environ["TEST_API_KEY"] = "sk-123456"
            result = EnvService.substitute_env_vars("Bearer ${TEST_API_KEY}")
            assert result == "Bearer sk-123456"
        finally:
            if original:
                os.environ["TEST_API_KEY"] = original
            else:
                os.environ.pop("TEST_API_KEY", None)

    def test_substitute_missing_var_returns_empty(self):
        """Test that missing var without default returns empty string."""
        result = EnvService.substitute_env_vars("${NONEXISTENT_VAR_12345}")
        assert result == ""

    def test_substitute_no_vars_passes_through(self):
        """Test string without vars passes through unchanged."""
        result = EnvService.substitute_env_vars("plain string")
        assert result == "plain string"

    # ==========================================================================
    # ${VAR:-default} syntax
    # ==========================================================================

    def test_substitute_with_default_when_missing(self):
        """Test ${VAR:-default} uses default when var is missing."""
        result = EnvService.substitute_env_vars("${MISSING_VAR:-fallback_value}")
        assert result == "fallback_value"

    def test_substitute_with_default_when_set(self):
        """Test ${VAR:-default} uses var value when set."""
        original = os.environ.get("TEST_VAR_DEFAULT")
        try:
            os.environ["TEST_VAR_DEFAULT"] = "actual_value"
            result = EnvService.substitute_env_vars("${TEST_VAR_DEFAULT:-fallback}")
            assert result == "actual_value"
        finally:
            if original:
                os.environ["TEST_VAR_DEFAULT"] = original
            else:
                os.environ.pop("TEST_VAR_DEFAULT", None)

    def test_substitute_with_empty_default(self):
        """Test ${VAR:-} returns empty string when missing."""
        result = EnvService.substitute_env_vars("${MISSING:-}")
        assert result == ""

    def test_substitute_default_with_special_chars(self):
        """Test default value can contain URLs and special chars."""
        result = EnvService.substitute_env_vars(
            "${MISSING:-https://api.example.com/v1}"
        )
        assert result == "https://api.example.com/v1"

    # ==========================================================================
    # ${file:filename:VAR} explicit file syntax
    # ==========================================================================

    def test_substitute_from_explicit_file(self):
        """Test ${file:.env.local:VAR} loads from specific file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .env.local with a variable
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_text("MY_SECRET=from_local_file\n")

            result = EnvService.substitute_env_vars(
                "${file:.env.local:MY_SECRET}", project_dir=tmpdir
            )
            assert result == "from_local_file"

    def test_substitute_from_explicit_file_with_default(self):
        """Test ${file:.env.local:VAR:-default} with default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # File doesn't exist, should use default
            result = EnvService.substitute_env_vars(
                "${file:.env.local:MISSING:-default_val}", project_dir=tmpdir
            )
            assert result == "default_val"

    def test_substitute_different_files(self):
        """Test different ${file:} references in same string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two different env files
            (Path(tmpdir) / ".env.a").write_text("VAR_A=value_a\n")
            (Path(tmpdir) / ".env.b").write_text("VAR_B=value_b\n")

            result = EnvService.substitute_env_vars(
                "${file:.env.a:VAR_A} and ${file:.env.b:VAR_B}", project_dir=tmpdir
            )
            assert result == "value_a and value_b"

    # ==========================================================================
    # Default .env file loading
    # ==========================================================================

    def test_substitute_from_default_env_file(self):
        """Test ${VAR} loads from .env in project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .env with a variable
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("PROJECT_VAR=from_dotenv\n")

            result = EnvService.substitute_env_vars(
                "${PROJECT_VAR}", project_dir=tmpdir
            )
            assert result == "from_dotenv"

    def test_env_file_takes_precedence_over_environ(self):
        """Test .env file values take precedence over os.environ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.environ.get("PRIORITY_TEST")
            try:
                os.environ["PRIORITY_TEST"] = "from_environ"
                env_path = Path(tmpdir) / ".env"
                env_path.write_text("PRIORITY_TEST=from_dotenv\n")

                result = EnvService.substitute_env_vars(
                    "${PRIORITY_TEST}", project_dir=tmpdir
                )
                assert result == "from_dotenv"
            finally:
                if original:
                    os.environ["PRIORITY_TEST"] = original
                else:
                    os.environ.pop("PRIORITY_TEST", None)

    def test_fallback_to_environ_when_not_in_file(self):
        """Test falls back to os.environ when not in .env file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original = os.environ.get("FALLBACK_VAR")
            try:
                os.environ["FALLBACK_VAR"] = "from_environ"
                # Empty .env file
                env_path = Path(tmpdir) / ".env"
                env_path.write_text("OTHER_VAR=other\n")

                result = EnvService.substitute_env_vars(
                    "${FALLBACK_VAR}", project_dir=tmpdir
                )
                assert result == "from_environ"
            finally:
                if original:
                    os.environ["FALLBACK_VAR"] = original
                else:
                    os.environ.pop("FALLBACK_VAR", None)

    # ==========================================================================
    # Dict substitution
    # ==========================================================================

    def test_substitute_in_dict_simple(self):
        """Test substitution in dict values."""
        original = os.environ.get("DICT_VAR")
        try:
            os.environ["DICT_VAR"] = "dict_value"
            data = {"key": "${DICT_VAR}", "static": "unchanged"}
            result = EnvService.substitute_in_dict(data)
            assert result == {"key": "dict_value", "static": "unchanged"}
        finally:
            if original:
                os.environ["DICT_VAR"] = original
            else:
                os.environ.pop("DICT_VAR", None)

    def test_substitute_in_dict_nested(self):
        """Test substitution in nested dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("NESTED_VAR=nested_value\n")

            data = {
                "runtime": {"models": [{"name": "test", "api_key": "${NESTED_VAR}"}]}
            }
            result = EnvService.substitute_in_dict(data, project_dir=tmpdir)
            assert result["runtime"]["models"][0]["api_key"] == "nested_value"

    def test_substitute_in_dict_list(self):
        """Test substitution in list values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("LIST_VAR=list_val\n")

            data = {"items": ["${LIST_VAR}", "static"]}
            result = EnvService.substitute_in_dict(data, project_dir=tmpdir)
            assert result["items"] == ["list_val", "static"]

    def test_substitute_in_dict_preserves_non_strings(self):
        """Test non-string values are preserved."""
        data = {"num": 42, "bool": True, "float": 3.14, "none": None}
        result = EnvService.substitute_in_dict(data)
        assert result == data

    # ==========================================================================
    # Utility methods
    # ==========================================================================

    def test_has_env_vars_true(self):
        """Test has_env_vars returns True for strings with vars."""
        assert EnvService.has_env_vars("${VAR}") is True
        assert EnvService.has_env_vars("prefix ${VAR} suffix") is True
        assert EnvService.has_env_vars("${file:.env:VAR}") is True

    def test_has_env_vars_false(self):
        """Test has_env_vars returns False for strings without vars."""
        assert EnvService.has_env_vars("plain") is False
        assert EnvService.has_env_vars("$NOT_A_VAR") is False
        assert EnvService.has_env_vars("{NOT_A_VAR}") is False

    def test_find_env_vars(self):
        """Test find_env_vars returns var info."""
        result = EnvService.find_env_vars("${VAR1} and ${file:.env:VAR2}")
        assert result == [(None, "VAR1"), (".env", "VAR2")]

    def test_find_env_vars_empty(self):
        """Test find_env_vars returns empty for no vars."""
        result = EnvService.find_env_vars("no variables")
        assert result == []

    # ==========================================================================
    # Cache behavior
    # ==========================================================================

    def test_file_cache_works(self):
        """Test that file contents are cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("CACHE_VAR=initial\n")

            # First read
            result1 = EnvService.substitute_env_vars("${CACHE_VAR}", project_dir=tmpdir)
            assert result1 == "initial"

            # Modify file (cache should still return old value)
            env_path.write_text("CACHE_VAR=modified\n")
            result2 = EnvService.substitute_env_vars("${CACHE_VAR}", project_dir=tmpdir)
            assert result2 == "initial"  # Still cached

            # Clear cache and re-read
            EnvService.clear_cache()
            result3 = EnvService.substitute_env_vars("${CACHE_VAR}", project_dir=tmpdir)
            assert result3 == "modified"  # Now updated

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_multiple_vars_in_string(self):
        """Test multiple ${VAR} in one string."""
        original_a = os.environ.get("MULTI_A")
        original_b = os.environ.get("MULTI_B")
        try:
            os.environ["MULTI_A"] = "val_a"
            os.environ["MULTI_B"] = "val_b"
            result = EnvService.substitute_env_vars("${MULTI_A}:${MULTI_B}")
            assert result == "val_a:val_b"
        finally:
            if original_a:
                os.environ["MULTI_A"] = original_a
            else:
                os.environ.pop("MULTI_A", None)
            if original_b:
                os.environ["MULTI_B"] = original_b
            else:
                os.environ.pop("MULTI_B", None)

    def test_var_name_with_underscore(self):
        """Test var names with underscores."""
        original = os.environ.get("MY_LONG_VAR_NAME")
        try:
            os.environ["MY_LONG_VAR_NAME"] = "value"
            result = EnvService.substitute_env_vars("${MY_LONG_VAR_NAME}")
            assert result == "value"
        finally:
            if original:
                os.environ["MY_LONG_VAR_NAME"] = original
            else:
                os.environ.pop("MY_LONG_VAR_NAME", None)

    def test_var_name_with_numbers(self):
        """Test var names with numbers."""
        original = os.environ.get("VAR123")
        try:
            os.environ["VAR123"] = "value"
            result = EnvService.substitute_env_vars("${VAR123}")
            assert result == "value"
        finally:
            if original:
                os.environ["VAR123"] = original
            else:
                os.environ.pop("VAR123", None)

    def test_missing_env_file_returns_empty_dict(self):
        """Test loading nonexistent env file returns empty dict."""
        result = EnvService.load_env_file(Path("/nonexistent/.env"))
        assert result == {}
