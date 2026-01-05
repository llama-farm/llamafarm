"""Tests for parser safety - preventing inappropriate fallbacks.

This module tests the fix for issue #589 where PDFs were being incorrectly
processed by a txt parser fallback, creating useless/garbage chunks.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.datamodel import DataProcessingStrategy, Parser

from core.blob_processor import BlobProcessor
from utils.parsing_safety import (
    ParserFailedError,
    ParsingError,
    UnsupportedFileTypeError,
    get_file_extension,
    is_binary_extension,
    validate_parser_for_file,
)


class TestBinaryExtensionDetection:
    """Test binary file extension detection."""

    def test_known_binary_extensions_detected(self):
        """Known binary extensions should be detected as binary."""
        binary_files = [
            "document.pdf",
            "file.docx",
            "data.xlsx",
            "image.png",
            "photo.jpg",
            "photo.jpeg",
            "archive.zip",
            "archive.tar",
            "archive.gz",
            "video.mp4",
            "audio.mp3",
            "email.msg",
            "binary.exe",
            "library.dll",
        ]
        for f in binary_files:
            assert is_binary_extension(f), f"{f} should be detected as binary"

    def test_text_extensions_not_binary(self):
        """Text extensions should not be detected as binary."""
        text_files = [
            "readme.txt",
            "code.py",
            "data.json",
            "config.yaml",
            "config.yml",
            "notes.md",
            "page.html",
            "script.js",
            "styles.css",
            "data.csv",
            "data.tsv",
            "document.xml",
        ]
        for f in text_files:
            assert not is_binary_extension(f), f"{f} should not be detected as binary"

    def test_case_insensitive_extension(self):
        """Extension detection should be case-insensitive."""
        assert is_binary_extension("FILE.PDF")
        assert is_binary_extension("file.Pdf")
        assert is_binary_extension("file.PDF")
        assert not is_binary_extension("FILE.TXT")
        assert not is_binary_extension("file.Txt")

    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("document.pdf") == ".pdf"
        assert get_file_extension("DOCUMENT.PDF") == ".pdf"
        assert get_file_extension("file.tar.gz") == ".gz"
        assert get_file_extension("no_extension") == ""
        # .hidden files are treated as filenames without extensions in Python's Path
        assert get_file_extension(".hidden") == ""
        assert get_file_extension(".hidden.txt") == ".txt"


class TestValidateParserForFile:
    """Test parser-file compatibility validation."""

    def test_text_parser_rejected_for_pdf(self):
        """TextParser should be rejected for PDF files."""
        is_valid, error = validate_parser_for_file("document.pdf", "TextParser_Python")
        assert not is_valid
        assert "Cannot use TextParser_Python" in error
        assert ".pdf" in error

    def test_text_parser_rejected_for_docx(self):
        """TextParser should be rejected for DOCX files."""
        is_valid, error = validate_parser_for_file("document.docx", "TextParser_Python")
        assert not is_valid
        assert "Cannot use TextParser_Python" in error

    def test_text_parser_accepted_for_txt(self):
        """TextParser should be accepted for TXT files."""
        is_valid, error = validate_parser_for_file("document.txt", "TextParser_Python")
        assert is_valid
        assert error is None

    def test_pdf_parser_accepted_for_pdf(self):
        """PDFParser should be accepted for PDF files."""
        is_valid, error = validate_parser_for_file("document.pdf", "PDFParser_PyPDF2")
        assert is_valid
        assert error is None


class TestParserExceptions:
    """Test parser exception classes."""

    def test_unsupported_file_type_error_attributes(self):
        """UnsupportedFileTypeError should have proper attributes."""
        error = UnsupportedFileTypeError(
            filename="document.pdf",
            extension=".pdf",
            available_parsers=["TextParser_Python", "MarkdownParser_Python"],
        )
        assert error.filename == "document.pdf"
        assert error.extension == ".pdf"
        assert "TextParser_Python" in error.available_parsers
        assert "document.pdf" in str(error)
        assert ".pdf" in str(error)

    def test_parser_failed_error_attributes(self):
        """ParserFailedError should have proper attributes."""
        error = ParserFailedError(
            filename="corrupt.pdf",
            tried_parsers=["PDFParser_PyPDF2", "PDFParser_LlamaIndex"],
            errors=["Error 1", "Error 2"],
        )
        assert error.filename == "corrupt.pdf"
        assert len(error.tried_parsers) == 2
        assert len(error.errors) == 2
        assert "corrupt.pdf" in str(error)
        assert "PDFParser_PyPDF2" in str(error)

    def test_exception_hierarchy(self):
        """Parser exceptions should inherit from ParsingError."""
        assert issubclass(UnsupportedFileTypeError, ParsingError)
        assert issubclass(ParserFailedError, ParsingError)


class TestBlobProcessorFailFast:
    """Test BlobProcessor fail-fast behavior."""

    def test_pdf_without_pdf_parser_raises_error_fail_fast_true(self):
        """PDF file with only text parser configured should raise error when fail_fast=True."""
        strategy = DataProcessingStrategy(
            name="text_only_strategy",
            description="Only has text parser configured for testing",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],  # Only matches txt files
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        # Fake PDF content (PDF magic bytes)
        pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(pdf_bytes, {"filename": "document.pdf"})

        assert "document.pdf" in str(exc_info.value)
        assert ".pdf" in exc_info.value.extension

    def test_pdf_without_parser_legacy_mode_returns_empty(self):
        """PDF file with fail_fast=False should return empty list."""
        strategy = DataProcessingStrategy(
            name="text_only_strategy",
            description="Only has text parser configured for testing",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],  # Only matches txt files
                )
            ],
            fail_fast=False,  # Legacy mode
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        result = processor.process_blob(pdf_bytes, {"filename": "document.pdf"})

        # Should return empty list, not garbage chunks
        assert result == []

    def test_txt_file_can_use_text_parser_fallback(self):
        """Text file should still work with text parser fallback."""
        strategy = DataProcessingStrategy(
            name="text_strategy",
            description="Text parser without include patterns",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    # No file_include_patterns - accepts all non-binary files
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        txt_bytes = b"Hello, world! This is a test file."
        result = processor.process_blob(txt_bytes, {"filename": "hello.txt"})

        assert len(result) > 0
        assert "Hello, world!" in result[0].content

    def test_docx_without_docx_parser_raises_error(self):
        """DOCX file with only text parser should raise error."""
        strategy = DataProcessingStrategy(
            name="text_only_strategy",
            description="Only has text parser configured for testing",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        # DOCX is actually a ZIP file
        docx_bytes = b"PK\x03\x04\x14\x00\x06\x00"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(docx_bytes, {"filename": "document.docx"})

        assert ".docx" in exc_info.value.extension

    def test_unknown_text_extension_can_use_text_parser(self):
        """Unknown text-like extensions should still work with text parser."""
        strategy = DataProcessingStrategy(
            name="text_strategy",
            description="Text parser without include patterns",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        content = b"Some custom file content"
        result = processor.process_blob(content, {"filename": "file.customext"})

        # Unknown non-binary extension should work with text parser
        assert len(result) > 0

    def test_fail_fast_default_is_true(self):
        """When fail_fast is not specified, it should default to True."""
        strategy = DataProcessingStrategy(
            name="text_only_strategy",
            description="Only has text parser configured for testing",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],
                )
            ],
            # fail_fast not specified - should default to True
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n"

        # Should raise error because fail_fast defaults to True
        with pytest.raises(UnsupportedFileTypeError):
            processor.process_blob(pdf_bytes, {"filename": "document.pdf"})

    @patch("core.blob_processor.BlobProcessor._get_parser_class")
    def test_all_parsers_fail_raises_parser_failed_error(self, mock_get_parser_class):
        """When all configured parsers fail, should raise ParserFailedError."""

        # Create a mock parser class that always fails
        class FailingParser:
            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                raise ValueError("Parser intentionally failed for testing")

        mock_get_parser_class.return_value = FailingParser

        strategy = DataProcessingStrategy(
            name="failing_strategy",
            description="Strategy with parser that always fails",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    config={},
                    file_include_patterns=["*.pdf"],
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n"

        with pytest.raises(ParserFailedError) as exc_info:
            processor.process_blob(pdf_bytes, {"filename": "test.pdf"})

        assert "test.pdf" in str(exc_info.value)
        assert "PDFParser_PyPDF2" in exc_info.value.tried_parsers

    @patch("core.blob_processor.BlobProcessor._get_parser_class")
    def test_all_parsers_fail_legacy_mode_returns_empty(self, mock_get_parser_class):
        """When all parsers fail with fail_fast=False, should return empty list."""

        class FailingParser:
            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                raise ValueError("Parser intentionally failed for testing")

        mock_get_parser_class.return_value = FailingParser

        strategy = DataProcessingStrategy(
            name="failing_strategy",
            description="Strategy with parser that always fails",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    config={},
                    file_include_patterns=["*.pdf"],
                )
            ],
            fail_fast=False,
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n"
        result = processor.process_blob(pdf_bytes, {"filename": "test.pdf"})

        # Should return empty list, not garbage chunks
        assert result == []


class TestImageFileSafety:
    """Test that image files are properly rejected."""

    @pytest.mark.parametrize(
        "extension",
        [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"],
    )
    def test_image_extensions_are_binary(self, extension):
        """Image file extensions should be detected as binary."""
        assert is_binary_extension(f"photo{extension}")

    def test_image_without_parser_raises_error(self):
        """Image file without image parser should raise error."""
        strategy = DataProcessingStrategy(
            name="text_only",
            description="Only text parser for testing purposes",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],
                )
            ],
            fail_fast=True,
        )
        processor = BlobProcessor(strategy)

        # PNG file header
        png_bytes = b"\x89PNG\r\n\x1a\n"

        with pytest.raises(UnsupportedFileTypeError):
            processor.process_blob(png_bytes, {"filename": "image.png"})


class TestArchiveFileSafety:
    """Test that archive files are properly rejected."""

    @pytest.mark.parametrize(
        "extension",
        [".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"],
    )
    def test_archive_extensions_are_binary(self, extension):
        """Archive file extensions should be detected as binary."""
        assert is_binary_extension(f"archive{extension}")

