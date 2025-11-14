"""Tests for preprocessor components."""

import tempfile
from pathlib import Path

import pytest

from components.preprocessors.base import BasePreprocessor, PreprocessorResult
from components.preprocessors.factory import PreprocessorFactory


class TestPreprocessorFactory:
    """Test PreprocessorFactory functionality."""

    def test_discover_preprocessors(self):
        """Test preprocessor discovery."""
        preprocessors = PreprocessorFactory.discover_preprocessors()
        assert isinstance(preprocessors, dict)
        # Should have at least markitdown and ocr types
        assert len(preprocessors) >= 2

    def test_list_preprocessors(self):
        """Test listing all preprocessors."""
        preprocessor_names = PreprocessorFactory.list_preprocessors()
        assert isinstance(preprocessor_names, list)
        assert "MarkItDownPreprocessor" in preprocessor_names
        # OCR preprocessors should be listed
        assert "PaddleOCRPreprocessor" in preprocessor_names

    def test_get_preprocessor_info(self):
        """Test getting preprocessor information."""
        info = PreprocessorFactory.get_preprocessor_info("MarkItDownPreprocessor")
        assert info is not None
        assert info["name"] == "MarkItDownPreprocessor"
        assert "supported_extensions" in info
        assert "dependencies" in info

    def test_load_preprocessor_class(self):
        """Test loading a preprocessor class."""
        preprocessor_class = PreprocessorFactory.load_preprocessor_class(
            "MarkItDownPreprocessor"
        )
        assert preprocessor_class is not None
        assert issubclass(preprocessor_class, BasePreprocessor)

    def test_create_preprocessor(self):
        """Test creating a preprocessor instance."""
        preprocessor = PreprocessorFactory.create(
            "MarkItDownPreprocessor", config={}
        )
        assert preprocessor is not None
        assert isinstance(preprocessor, BasePreprocessor)


class TestMarkItDownPreprocessor:
    """Test MarkItDown preprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create MarkItDown preprocessor instance."""
        try:
            return PreprocessorFactory.create("MarkItDownPreprocessor", config={})
        except ImportError:
            pytest.skip("MarkItDown not installed")

    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initializes correctly."""
        assert preprocessor is not None
        assert hasattr(preprocessor, "can_process")
        assert hasattr(preprocessor, "preprocess")
        assert hasattr(preprocessor, "get_supported_formats")

    def test_can_process_supported_formats(self, preprocessor):
        """Test can_process returns True for supported formats."""
        # Test various supported formats (lowercase)
        assert preprocessor.can_process("test.docx", {})
        assert preprocessor.can_process("test.pdf", {})
        assert preprocessor.can_process("test.pptx", {})
        assert preprocessor.can_process("test.html", {})

        # Test uppercase extensions
        assert preprocessor.can_process("test.DOCX", {})
        assert preprocessor.can_process("test.PDF", {})
        assert preprocessor.can_process("test.PPTX", {})
        assert preprocessor.can_process("test.HTML", {})

        # Test filenames with multiple dots
        assert preprocessor.can_process("my.file.name.docx", {})
        assert preprocessor.can_process("archive.v1.2.pdf", {})
        assert preprocessor.can_process("presentation.final.pptx", {})
        assert preprocessor.can_process("web.page.html", {})

    def test_can_process_unsupported_formats(self, preprocessor):
        """Test can_process returns False for unsupported formats."""
        assert not preprocessor.can_process("test.exe", {})
        assert not preprocessor.can_process("test.unknown", {})

        # Test filenames with no extension
        assert not preprocessor.can_process("noextension", {})
        assert not preprocessor.can_process("anotherfile", {})

    def test_get_supported_formats(self, preprocessor):
        """Test getting supported formats."""
        formats = preprocessor.get_supported_formats()
        assert isinstance(formats, list)
        assert ".docx" in formats
        assert ".pdf" in formats
        assert ".pptx" in formats

    def test_preprocess_text_file(self, preprocessor):
        """Test preprocessing a simple text file (via HTML wrapper)."""
        # Create a temporary HTML file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False
        ) as tmp:
            tmp.write("<html><body><h1>Test</h1><p>This is a test.</p></body></html>")
            tmp.flush()
            tmp_path = tmp.name

        try:
            result = preprocessor.preprocess(tmp_path, {"filename": "test.html"})

            assert isinstance(result, PreprocessorResult)
            assert result.success
            assert len(result.content) > 0
            assert "Test" in result.content
            assert result.output_format == "markdown"
            assert result.metadata["preprocessor"] == "MarkItDownPreprocessor"

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_preprocess_nonexistent_file(self, preprocessor):
        """Test preprocessing a file that doesn't exist."""
        result = preprocessor.preprocess(
            "/tmp/nonexistent_file.docx", {"filename": "nonexistent.docx"}
        )

        assert isinstance(result, PreprocessorResult)
        assert not result.success
        assert len(result.errors) > 0


class TestOCRPreprocessor:
    """Test OCR preprocessor base functionality."""

    @pytest.fixture
    def preprocessor(self):
        """Create OCR preprocessor instance."""
        try:
            return PreprocessorFactory.create("PaddleOCRPreprocessor", config={})
        except ImportError:
            pytest.skip("PaddleOCR not installed")

    def test_preprocessor_initialization(self, preprocessor):
        """Test OCR preprocessor initializes correctly."""
        assert preprocessor is not None
        assert hasattr(preprocessor, "can_process")
        assert hasattr(preprocessor, "preprocess")

    def test_can_process_images(self, preprocessor):
        """Test can_process returns True for image formats."""
        assert preprocessor.can_process("test.png", {})
        assert preprocessor.can_process("test.jpg", {})
        assert preprocessor.can_process("test.jpeg", {})

    def test_can_process_unsupported_formats(self, preprocessor):
        """Test can_process returns False for non-image formats."""
        assert not preprocessor.can_process("test.docx", {})
        assert not preprocessor.can_process("test.txt", {})

    def test_get_supported_formats(self, preprocessor):
        """Test getting supported formats."""
        formats = preprocessor.get_supported_formats()
        assert isinstance(formats, list)
        assert ".png" in formats
        assert ".jpg" in formats
        assert ".pdf" in formats

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        from components.preprocessors.ocr.paddleocr_preprocessor import (
            PaddleOCRPreprocessor,
        )

        with pytest.raises(ValueError) as excinfo:
            PaddleOCRPreprocessor(config={"language": "invalid_lang"})

        assert "not supported by PaddleOCR" in str(excinfo.value)
        assert "invalid_lang" in str(excinfo.value)


class TestPreprocessorResult:
    """Test PreprocessorResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        result = PreprocessorResult(
            content="Test content",
            metadata={"key": "value"},
            output_format="markdown",
            success=True,
        )

        assert result.success
        assert result.content == "Test content"
        assert result.metadata["key"] == "value"
        assert result.output_format == "markdown"
        assert len(result.errors) == 0

    def test_failed_result(self):
        """Test creating a failed result."""
        result = PreprocessorResult(
            content="",
            metadata={},
            output_format="text",
            success=False,
            errors=["Error message"],
        )

        assert not result.success
        assert result.content == ""
        assert len(result.errors) == 1
        assert result.errors[0] == "Error message"

    def test_result_with_output_file(self):
        """Test result with output file path."""
        result = PreprocessorResult(
            content="Content",
            metadata={},
            output_format="searchable_pdf",
            output_file="/path/to/output.pdf",
            success=True,
        )

        assert result.success
        assert result.output_file == "/path/to/output.pdf"
        assert result.output_format == "searchable_pdf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
