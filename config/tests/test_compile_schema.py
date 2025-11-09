#!/usr/bin/env python3
"""
Tests for compile_schema.py, focusing on cross-platform URI handling.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from compile_schema import load_text_from_uri


class TestURIHandling:
    """Test URI handling across different platforms and formats."""

    def test_plain_path_loading(self, tmp_path):
        """Test loading from a plain filesystem path (no file:// scheme)."""
        test_file = tmp_path / "test.yaml"
        test_content = "test: value\n"
        test_file.write_text(test_content, encoding="utf-8")

        result = load_text_from_uri(str(test_file))
        assert result == test_content

    def test_file_uri_unix_style(self, tmp_path):
        """Test loading from a file:// URI with Unix-style paths."""
        test_file = tmp_path / "test.yaml"
        test_content = "test: unix\n"
        test_file.write_text(test_content, encoding="utf-8")

        # Convert to file URI (Unix style)
        file_uri = test_file.as_uri()
        assert file_uri.startswith("file://")

        result = load_text_from_uri(file_uri)
        assert result == test_content

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_file_uri_windows_drive_letter(self, tmp_path):
        """Test loading from a file:// URI with Windows drive letters (C:, D:, etc)."""
        test_file = tmp_path / "test.yaml"
        test_content = "test: windows\n"
        test_file.write_text(test_content, encoding="utf-8")

        # Convert to file URI (Windows style, e.g., file:///C:/Users/...)
        file_uri = test_file.as_uri()
        assert file_uri.startswith("file:///")
        assert ":" in file_uri[8:]  # Drive letter colon after file:///

        result = load_text_from_uri(file_uri)
        assert result == test_content

    def test_file_uri_with_spaces(self, tmp_path):
        """Test loading from a file:// URI with spaces (URL-encoded)."""
        test_dir = tmp_path / "dir with spaces"
        test_dir.mkdir()
        test_file = test_dir / "test file.yaml"
        test_content = "test: spaces\n"
        test_file.write_text(test_content, encoding="utf-8")

        file_uri = test_file.as_uri()
        assert "%20" in file_uri  # Spaces should be URL-encoded

        result = load_text_from_uri(file_uri)
        assert result == test_content

    def test_invalid_uri_scheme_raises_error(self):
        """Test that non-file URI schemes raise an error."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            load_text_from_uri("http://example.com/test.yaml")

        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            load_text_from_uri("ftp://example.com/test.yaml")

    def test_nonexistent_file_raises_error(self):
        """Test that loading a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            load_text_from_uri("/nonexistent/path/to/file.yaml")

    @patch("sys.platform", "win32")
    def test_windows_path_conversion(self, tmp_path):
        """Test that Windows file URIs are properly converted to filesystem paths."""
        # Simulate a Windows file URI structure
        # On Windows, Path.as_uri() returns file:///C:/Users/...
        # urlparse gives parsed.path = '/C:/Users/...'
        # url2pathname should convert this to 'C:\\Users\\...'

        test_file = tmp_path / "test.yaml"
        test_content = "test: win_conversion\n"
        test_file.write_text(test_content, encoding="utf-8")

        # Get the actual file URI (platform-specific)
        file_uri = test_file.as_uri()

        # This should work on any platform thanks to url2pathname
        result = load_text_from_uri(file_uri)
        assert result == test_content

    def test_relative_path_resolution(self, tmp_path):
        """Test that relative paths are handled correctly."""
        test_file = tmp_path / "relative.yaml"
        test_content = "test: relative\n"
        test_file.write_text(test_content, encoding="utf-8")

        # Change to the temp directory
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = load_text_from_uri("relative.yaml")
            assert result == test_content
        finally:
            os.chdir(original_cwd)


class TestSchemaCompilation:
    """Test the full schema compilation process."""

    def test_schema_compilation_succeeds(self):
        """Test that the main schema can be compiled without errors."""
        from compile_schema import load_and_deref_schema

        schema_path = Path(__file__).parent.parent / "schema.yaml"
        if schema_path.exists():
            # This should not raise any errors
            schema = load_and_deref_schema(schema_path)
            assert isinstance(schema, dict)
            assert "$schema" in schema or "type" in schema

    def test_schema_with_external_ref(self, tmp_path):
        """Test schema compilation with external $ref references."""
        from compile_schema import load_and_deref_schema

        # Create a referenced schema
        ref_schema = tmp_path / "referenced.yaml"
        ref_schema.write_text(
            """
type: object
properties:
  name:
    type: string
""",
            encoding="utf-8",
        )

        # Create a main schema that references it
        main_schema = tmp_path / "main.yaml"
        main_schema.write_text(
            f"""
type: object
properties:
  referenced:
    $ref: "{ref_schema.as_uri()}"
""",
            encoding="utf-8",
        )

        # Load and dereference
        result = load_and_deref_schema(main_schema)
        assert isinstance(result, dict)
        assert "properties" in result
        assert "referenced" in result["properties"]
        # The $ref should be resolved
        assert "properties" in result["properties"]["referenced"]
        assert "name" in result["properties"]["referenced"]["properties"]


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
