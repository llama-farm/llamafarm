"""Tests for PDF parser."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from components.parsers.pdf_parser import PDFParser
from core.base import Document


class TestPDFParser:
    """Test the PDF parser."""

    def test_parser_initialization(self):
        """Test parser initialization with config."""
        config = {
            "extract_metadata": True,
            "extract_page_structure": True,
            "combine_pages": False,
            "page_separator": "\n--- PAGE ---\n",
            "min_text_length": 5,
            "include_page_numbers": False,
            "extract_outline": True,
        }

        parser = PDFParser(config=config)

        assert parser.extract_metadata is True
        assert parser.extract_page_structure is True
        assert parser.combine_pages is False
        assert parser.page_separator == "\n--- PAGE ---\n"
        assert parser.min_text_length == 5
        assert parser.include_page_numbers is False
        assert parser.extract_outline is True

    def test_parser_default_config(self):
        """Test parser with default configuration."""
        parser = PDFParser()

        assert parser.extract_metadata is True
        assert parser.extract_page_structure is True
        assert parser.combine_pages is True
        assert parser.page_separator == "\n\n--- Page Break ---\n\n"
        assert parser.min_text_length == 10
        assert parser.include_page_numbers is True
        assert parser.extract_outline is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        parser = PDFParser(config={"min_text_length": 0})
        assert parser.validate_config() is True

        # Invalid config
        with pytest.raises(ValueError, match="min_text_length must be non-negative"):
            parser = PDFParser(config={"min_text_length": -1})
            parser.validate_config()

    def test_parse_nonexistent_file(self):
        """Test parsing of non-existent file."""
        parser = PDFParser()
        result = parser.parse("nonexistent_file.pdf")

        assert len(result.documents) == 0
        assert len(result.errors) > 0
        assert "not found" in result.errors[0]["error"].lower()

    @pytest.mark.skipif(
        not Path("samples/pdfs/test_document.pdf").exists(),
        reason="Test PDF not available",
    )
    def test_parse_real_pdf(self):
        """Test parsing a real PDF file."""
        parser = PDFParser()
        result = parser.parse("samples/pdfs/test_document.pdf")

        # Should successfully parse the PDF
        assert len(result.documents) > 0
        assert len(result.errors) == 0

        doc = result.documents[0]
        assert isinstance(doc, Document)
        assert len(doc.content) > 0
        assert doc.source == "samples/pdfs/test_document.pdf"
        assert "parser_type" in doc.metadata
        assert doc.metadata["parser_type"] == "PDFParser"

    @pytest.mark.skipif(
        not Path("samples/pdfs/test_document.pdf").exists(),
        reason="Test PDF not available",
    )
    def test_parse_pdf_separate_pages(self):
        """Test parsing PDF with separate page documents."""
        parser = PDFParser(config={"combine_pages": False})
        result = parser.parse("samples/pdfs/test_document.pdf")

        # Should create separate documents for each page
        assert len(result.documents) > 1  # Multi-page PDF

        for i, doc in enumerate(result.documents):
            assert f"_page_{i + 1}" in doc.id
            assert doc.metadata["page_number"] == i + 1
            assert "total_pages" in doc.metadata

    @pytest.mark.skipif(
        not Path("samples/pdfs/test_document.pdf").exists(),
        reason="Test PDF not available",
    )
    def test_parse_pdf_combined_pages(self):
        """Test parsing PDF with combined page content."""
        parser = PDFParser(config={"combine_pages": True})
        result = parser.parse("samples/pdfs/test_document.pdf")

        # Should create single document with all pages
        assert len(result.documents) == 1

        doc = result.documents[0]
        assert "total_pages" in doc.metadata
        assert doc.metadata["total_pages"] > 1
        assert "--- Page Break ---" in doc.content or "[Page " in doc.content

    @pytest.mark.skipif(
        not Path("samples/pdfs/test_document.pdf").exists(),
        reason="Test PDF not available",
    )
    def test_pdf_metadata_extraction(self):
        """Test PDF metadata extraction."""
        parser = PDFParser(config={"extract_metadata": True})
        result = parser.parse("samples/pdfs/test_document.pdf")

        assert len(result.documents) > 0
        doc = result.documents[0]

        # Check for expected metadata fields
        expected_fields = [
            "source_file",
            "file_size_bytes",
            "total_pages",
            "parser_type",
        ]
        for field in expected_fields:
            assert field in doc.metadata

        # Check specific values
        assert doc.metadata["source_file"] == "test_document.pdf"
        assert doc.metadata["file_size_bytes"] > 0
        assert doc.metadata["total_pages"] >= 1
        assert doc.metadata["parser_type"] == "PDFParser"

    def test_pdf_metadata_disabled(self):
        """Test PDF parsing with metadata extraction disabled."""
        parser = PDFParser(config={"extract_metadata": False})

        # Mock PyPDF2 for this test
        with patch("components.parsers.pdf_parser.pdf_parser.PyPDF2") as mock_pypdf2:
            mock_reader = MagicMock()
            mock_reader.pages = [MagicMock()]
            mock_reader.pages[0].extract_text.return_value = "Sample text content"
            mock_reader.metadata = {"Title": "Test Title"}
            mock_reader.is_encrypted = False
            mock_reader.outline = None
            mock_pypdf2.PdfReader.return_value = mock_reader

            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", delete=False
            ) as f:
                f.write(b"fake pdf content")
                temp_file = f.name

            try:
                result = parser.parse(temp_file)
                assert len(result.documents) > 0

                doc = result.documents[0]
                # Should have basic metadata but not PDF-specific fields
                assert "source_file" in doc.metadata
                assert "parser_type" in doc.metadata
                assert "title" not in doc.metadata  # PDF metadata disabled

            finally:
                Path(temp_file).unlink()

    def test_min_text_length_filtering(self):
        """Test filtering of pages with insufficient text."""
        parser = PDFParser(config={"min_text_length": 50})

        with patch("components.parsers.pdf_parser.pdf_parser.PyPDF2") as mock_pypdf2:
            mock_reader = MagicMock()
            # Create pages with different text lengths
            mock_page1 = MagicMock()
            mock_page1.extract_text.return_value = "Short"  # Too short
            mock_page2 = MagicMock()
            mock_page2.extract_text.return_value = (
                "This is a much longer text that exceeds the minimum length requirement"
            )

            mock_reader.pages = [mock_page1, mock_page2]
            mock_reader.metadata = None
            mock_reader.is_encrypted = False
            mock_reader.outline = None
            mock_pypdf2.PdfReader.return_value = mock_reader

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", delete=False
            ) as f:
                f.write(b"fake pdf content")
                temp_file = f.name

            try:
                result = parser.parse(temp_file)

                # Should only include the page with sufficient text
                assert len(result.documents) == 1
                doc = result.documents[0]
                assert "longer text" in doc.content

            finally:
                Path(temp_file).unlink()

    def test_page_number_inclusion(self):
        """Test inclusion of page numbers in text."""
        parser = PDFParser(
            config={"include_page_numbers": True, "combine_pages": False}
        )

        with patch("components.parsers.pdf_parser.pdf_parser.PyPDF2") as mock_pypdf2:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Sample page content"
            mock_reader.pages = [mock_page]
            mock_reader.metadata = None
            mock_reader.is_encrypted = False
            mock_reader.outline = None
            mock_pypdf2.PdfReader.return_value = mock_reader

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", delete=False
            ) as f:
                f.write(b"fake pdf content")
                temp_file = f.name

            try:
                result = parser.parse(temp_file)
                assert len(result.documents) > 0

                doc = result.documents[0]
                assert "[Page 1]" in doc.content

            finally:
                Path(temp_file).unlink()

    def test_outline_extraction(self):
        """Test PDF outline/bookmark extraction."""
        parser = PDFParser(config={"extract_outline": True})

        with patch("components.parsers.pdf_parser.pdf_parser.PyPDF2") as mock_pypdf2:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Sample content"
            mock_reader.pages = [mock_page]
            mock_reader.metadata = None
            mock_reader.is_encrypted = False

            # Mock outline structure
            mock_outline_item = MagicMock()
            mock_outline_item.title = "Chapter 1"
            mock_reader.outline = [mock_outline_item]

            mock_pypdf2.PdfReader.return_value = mock_reader

            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", delete=False
            ) as f:
                f.write(b"fake pdf content")
                temp_file = f.name

            try:
                result = parser.parse(temp_file)
                assert len(result.documents) > 0

                doc = result.documents[0]
                assert "Document Outline:" in doc.content
                assert "Chapter 1" in doc.content

            finally:
                Path(temp_file).unlink()
