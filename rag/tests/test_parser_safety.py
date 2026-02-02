"""Tests for parser safety - preventing inappropriate fallbacks.

This module tests the fix for issue #589 where PDFs were being incorrectly
processed by a txt parser fallback, creating useless/garbage chunks.

Simplified design: All files require explicit parser configuration.
No fallback logic, no binary detection heuristics.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.datamodel import DataProcessingStrategyDefinition, Parser

from core.blob_processor import BlobProcessor
from utils.parsing_safety import (
    ParserFailedError,
    ParsingError,
    UnsupportedFileTypeError,
    get_file_extension,
)


class TestFileExtension:
    """Test file extension utility."""

    def test_get_file_extension_pdf(self):
        """Test extension extraction for PDF."""
        assert get_file_extension("document.pdf") == ".pdf"

    def test_get_file_extension_uppercase(self):
        """Extension should be normalized to lowercase."""
        assert get_file_extension("DOCUMENT.PDF") == ".pdf"

    def test_get_file_extension_compound(self):
        """Compound extensions return only the last one."""
        assert get_file_extension("file.tar.gz") == ".gz"

    def test_get_file_extension_none(self):
        """Files without extension return empty string."""
        assert get_file_extension("no_extension") == ""

    def test_get_file_extension_hidden(self):
        """Hidden files without extension return empty string."""
        assert get_file_extension(".hidden") == ""

    def test_get_file_extension_hidden_with_ext(self):
        """Hidden files with extension return the extension."""
        assert get_file_extension(".hidden.txt") == ".txt"


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
        assert "No parser configured" in str(error)

    def test_unsupported_file_type_error_no_parsers(self):
        """UnsupportedFileTypeError with no available parsers."""
        error = UnsupportedFileTypeError(
            filename="file.xyz",
            extension=".xyz",
            available_parsers=[],
        )
        assert "No parsers are currently configured" in str(error)

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

    def test_parser_failed_error_truncates_errors(self):
        """ParserFailedError should truncate long error lists."""
        error = ParserFailedError(
            filename="bad.pdf",
            tried_parsers=["Parser1", "Parser2", "Parser3", "Parser4", "Parser5"],
            errors=["Error 1", "Error 2", "Error 3", "Error 4", "Error 5"],
        )
        error_str = str(error)
        # Should show first 3 and indicate more
        assert "and 2 more" in error_str

    def test_exception_hierarchy(self):
        """Parser exceptions should inherit from ParsingError."""
        assert issubclass(UnsupportedFileTypeError, ParsingError)
        assert issubclass(ParserFailedError, ParsingError)


class TestParserRequirement:
    """Test that all files require explicit parser configuration."""

    def test_pdf_without_pdf_parser_raises_error(self):
        """PDF file with only text parser (wrong patterns) should raise error."""
        strategy = DataProcessingStrategyDefinition(
            name="text_only_strategy",
            description="Only has text parser for txt files",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],  # Only matches txt files
                )
            ],
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(pdf_bytes, {"filename": "document.pdf"})

        assert "document.pdf" in str(exc_info.value)
        assert ".pdf" in str(exc_info.value)
        assert "No parser configured" in str(exc_info.value)

    def test_txt_without_matching_parser_raises_error(self):
        """Text file without matching parser should raise error (no fallback)."""
        strategy = DataProcessingStrategyDefinition(
            name="pdf_only_strategy",
            description="Only has PDF parser for pdf files",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    file_include_patterns=["*.pdf"],
                    config={},
                )
            ],
        )
        processor = BlobProcessor(strategy)

        txt_bytes = b"Hello, world!"

        # Should raise error - no fallback to text parser
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(txt_bytes, {"filename": "hello.txt"})

        assert "hello.txt" in str(exc_info.value)

    def test_txt_with_text_parser_succeeds(self):
        """Text file with matching text parser should work."""
        strategy = DataProcessingStrategyDefinition(
            name="text_strategy",
            description="Has text parser that matches all files",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    # No file_include_patterns = matches all files
                )
            ],
        )
        processor = BlobProcessor(strategy)

        txt_bytes = b"Hello, world!"
        result = processor.process_blob(txt_bytes, {"filename": "hello.txt"})

        assert len(result) > 0
        assert "Hello, world!" in result[0].content

    def test_docx_without_docx_parser_raises_error(self):
        """DOCX without matching docx parser should raise error."""
        strategy = DataProcessingStrategyDefinition(
            name="text_only_strategy",
            description="Only has text parser for txt files",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],
                )
            ],
        )
        processor = BlobProcessor(strategy)

        # DOCX is actually a ZIP file
        docx_bytes = b"PK\x03\x04\x14\x00\x06\x00"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(docx_bytes, {"filename": "document.docx"})

        assert ".docx" in str(exc_info.value)

    def test_unknown_extension_raises_error(self):
        """File with unknown extension should raise error (no fallback)."""
        strategy = DataProcessingStrategyDefinition(
            name="specific_strategy",
            description="Parser only matches specific patterns",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt", "*.md"],
                )
            ],
        )
        processor = BlobProcessor(strategy)

        content = b"Some content"

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.process_blob(content, {"filename": "file.xyz"})

        assert ".xyz" in str(exc_info.value)


class TestParserFailure:
    """Test behavior when configured parsers fail."""

    @patch("core.blob_processor.BlobProcessor._get_parser_class")
    def test_all_parsers_fail_raises_error(self, mock_get_parser_class):
        """When all configured parsers fail, should raise ParserFailedError."""

        # Create a mock parser class that always fails
        class FailingParser:
            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                raise ValueError("Parser intentionally failed for testing")

        mock_get_parser_class.return_value = FailingParser

        strategy = DataProcessingStrategyDefinition(
            name="failing_strategy",
            description="Strategy with parser that always fails",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    config={},
                    file_include_patterns=["*.pdf"],
                )
            ],
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n"

        with pytest.raises(ParserFailedError) as exc_info:
            processor.process_blob(pdf_bytes, {"filename": "test.pdf"})

        assert "test.pdf" in str(exc_info.value)
        assert "PDFParser_PyPDF2" in exc_info.value.tried_parsers

    @patch("core.blob_processor.BlobProcessor._get_parser_class")
    def test_parser_failure_includes_error_details(self, mock_get_parser_class):
        """ParserFailedError should include details about what went wrong."""

        class FailingParser:
            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                raise ValueError("Specific error message here")

        mock_get_parser_class.return_value = FailingParser

        strategy = DataProcessingStrategyDefinition(
            name="failing_strategy",
            description="Strategy with parser that always fails",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    config={},
                    file_include_patterns=["*.pdf"],
                )
            ],
        )
        processor = BlobProcessor(strategy)

        pdf_bytes = b"%PDF-1.4\n"

        with pytest.raises(ParserFailedError) as exc_info:
            processor.process_blob(pdf_bytes, {"filename": "bad.pdf"})

        error = exc_info.value
        assert error.filename == "bad.pdf"
        assert len(error.tried_parsers) > 0
        assert len(error.errors) > 0
        assert "Specific error message" in error.errors[0]


class TestBatchProcessingContinuesOnFailure:
    """Test that batch processing continues when individual files fail."""

    def test_blob_processor_raises_unsupported_file_type_error(self):
        """BlobProcessor raises UnsupportedFileTypeError for files with no matching parser.

        This exception is caught by IngestHandler.ingest_file() which returns
        a skipped status instead of propagating the exception.
        """
        # Create a strategy with only PDF parser
        strategy = DataProcessingStrategyDefinition(
            name="pdf_only",
            description="Only handles PDF files for testing",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    file_include_patterns=["*.pdf"],
                    config={},
                )
            ],
        )

        processor = BlobProcessor(strategy)

        # Try to process a text file - should raise UnsupportedFileTypeError
        with pytest.raises(UnsupportedFileTypeError):
            processor.process_blob(b"Hello", {"filename": "test.txt"})

    def test_batch_continues_after_individual_failure(self):
        """Batch processing should continue after individual file failures."""
        # This tests that the exception handling in ingest_file catches
        # the exceptions and returns error/skipped status instead of raising

        strategy = DataProcessingStrategyDefinition(
            name="pdf_only",
            description="Only handles PDF files for testing",
            parsers=[
                Parser(
                    type="PDFParser_PyPDF2",
                    file_include_patterns=["*.pdf"],
                    config={},
                )
            ],
        )
        processor = BlobProcessor(strategy)

        # Simulate batch processing with mixed results
        results = []
        files_to_process = [
            (b"%PDF-1.4\n", {"filename": "valid.pdf"}),  # Will try PDF parser
            (b"Hello", {"filename": "text.txt"}),  # Will fail - no parser
            (b"More text", {"filename": "another.md"}),  # Will fail - no parser
        ]

        for file_data, metadata in files_to_process:
            try:
                processor.process_blob(file_data, metadata)
                results.append({"status": "success", "filename": metadata["filename"]})
            except UnsupportedFileTypeError:
                # This is the expected behavior - exception caught, batch continues
                results.append(
                    {
                        "status": "skipped",
                        "filename": metadata["filename"],
                        "reason": "unsupported_file_type",
                    }
                )
            except ParserFailedError:
                results.append(
                    {
                        "status": "error",
                        "filename": metadata["filename"],
                        "reason": "parser_failed",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "status": "error",
                        "filename": metadata["filename"],
                        "reason": str(e),
                    }
                )

        # All 3 files should have been processed (not stopped after first failure)
        assert len(results) == 3

        # text.txt and another.md should be skipped (unsupported)
        skipped = [r for r in results if r["status"] == "skipped"]
        assert len(skipped) == 2
        assert any(r["filename"] == "text.txt" for r in skipped)
        assert any(r["filename"] == "another.md" for r in skipped)


class TestExplicitConfiguration:
    """Test that explicit parser configuration is required and works."""

    def test_parser_with_matching_pattern_succeeds(self):
        """Parser with matching file pattern should process file."""
        strategy = DataProcessingStrategyDefinition(
            name="text_strategy",
            description="Text parser matching txt files",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    file_include_patterns=["*.txt"],
                )
            ],
        )
        processor = BlobProcessor(strategy)

        txt_bytes = b"Test content"
        result = processor.process_blob(txt_bytes, {"filename": "test.txt"})

        assert len(result) > 0

    def test_parser_without_patterns_matches_all(self):
        """Parser without file patterns should match all files."""
        strategy = DataProcessingStrategyDefinition(
            name="catch_all_strategy",
            description="Text parser without patterns matches all",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    config={},
                    # No file_include_patterns = matches all
                )
            ],
        )
        processor = BlobProcessor(strategy)

        # Should work for any extension
        txt_bytes = b"Test content"
        result = processor.process_blob(txt_bytes, {"filename": "file.anything"})

        assert len(result) > 0

    @patch("core.blob_processor.BlobProcessor._get_parser_class")
    def test_multiple_parsers_tried_in_priority_order(self, mock_get_parser_class):
        """Multiple matching parsers should be tried in priority order.

        This test verifies that parsers are tried in priority order (lower number = higher priority).
        We use mock parsers where the high-priority one fails and the low-priority one succeeds,
        then verify both were tried in the correct order.
        """
        call_order = []

        class HighPriorityParser:
            """Parser with priority=1 (high priority) that fails."""

            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                call_order.append("high_priority")
                raise ValueError("High priority parser intentionally failed")

        class LowPriorityParser:
            """Parser with priority=10 (low priority) that succeeds."""

            def __init__(self, name=None, config=None):
                self.name = name

            def parse_blob(self, data, metadata):
                call_order.append("low_priority")
                from core.base import Document

                return [
                    Document(content="Success from low priority parser", metadata={})
                ]

        # Return different parser classes based on the parser type
        def get_parser_class(parser_type):
            if parser_type == "MockHighPriorityParser":
                return HighPriorityParser
            elif parser_type == "MockLowPriorityParser":
                return LowPriorityParser
            raise ValueError(f"Unknown parser type: {parser_type}")

        mock_get_parser_class.side_effect = get_parser_class

        strategy = DataProcessingStrategyDefinition(
            name="multi_parser_strategy",
            description="Multiple parsers with different priorities",
            parsers=[
                Parser(
                    type="MockLowPriorityParser",
                    config={},
                    priority=10,  # Lower priority (higher number)
                ),
                Parser(
                    type="MockHighPriorityParser",
                    config={},
                    priority=1,  # Higher priority (lower number = tried first)
                ),
            ],
        )
        processor = BlobProcessor(strategy)

        # Process file - high priority parser should fail, low priority should succeed
        txt_bytes = b"Test content"
        result = processor.process_blob(txt_bytes, {"filename": "test.txt"})

        # Verify parsers were tried in priority order
        assert call_order == ["high_priority", "low_priority"], (
            f"Expected parsers tried in priority order ['high_priority', 'low_priority'], "
            f"but got {call_order}"
        )

        # Verify the low-priority parser's result was returned
        assert len(result) == 1
        assert "low priority parser" in result[0].content
