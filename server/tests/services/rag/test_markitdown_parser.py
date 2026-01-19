"""Tests for MarkItDown parser with real files from examples/."""

import os
from pathlib import Path

import pytest

from services.rag.parsers.markitdown_parser import MarkItDownParser

# Get path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent.parent / "examples"
SAMPLE_FILES_DIR = EXAMPLES_DIR / "rag_pipeline" / "sample_files"


class TestMarkItDownParser:
    """Test MarkItDown parser with real files."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return MarkItDownParser()

    def test_parser_initialization(self, parser):
        """Test parser initializes correctly."""
        assert parser.converter is not None
        assert parser.ocr_endpoint == "http://127.0.0.1:11540/v1/ocr"

    def test_parse_pdf_real_file(self, parser):
        """Test parsing real PDF file."""
        pdf_file = SAMPLE_FILES_DIR / "Alpaca-Care-for-Beginners.pdf"
        assert pdf_file.exists(), f"Test file not found: {pdf_file}"

        result = parser.parse_sync(str(pdf_file))

        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)
        # Should extract some meaningful content
        assert len(result) > 100

    def test_parse_html_real_file(self, parser):
        """Test parsing real HTML file."""
        html_file = (
            SAMPLE_FILES_DIR / "news_articles" / "ai_breakthrough.html"
        )
        assert html_file.exists(), f"Test file not found: {html_file}"

        result = parser.parse_sync(str(html_file))

        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)

    def test_parse_fda_pdf(self, parser):
        """Test parsing FDA PDF document."""
        fda_file = SAMPLE_FILES_DIR / "fda" / "761248_2024_Orig1s000OtherActionLtrs.pdf"
        assert fda_file.exists(), f"Test file not found: {fda_file}"

        result = parser.parse_sync(str(fda_file))

        assert result is not None
        assert len(result) > 0
        # FDA documents should have substantial text
        assert len(result) > 50

    def test_needs_ocr_image_file(self, parser):
        """Test OCR detection for image files."""
        # Create temp file path with image extension
        assert parser._needs_ocr("test.png", "some text")
        assert parser._needs_ocr("test.jpg", "some text")
        assert parser._needs_ocr("test.jpeg", "some text")

    def test_needs_ocr_scanned_pdf(self, parser):
        """Test OCR detection for scanned PDFs with little text."""
        # PDF with very little text should trigger OCR
        assert parser._needs_ocr("test.pdf", "")
        assert parser._needs_ocr("test.pdf", "abc")
        assert parser._needs_ocr("test.pdf", "x" * 30)

        # PDF with sufficient text should NOT trigger OCR
        assert not parser._needs_ocr("test.pdf", "x" * 100)

    def test_needs_ocr_docx_no_trigger(self, parser):
        """Test that DOCX files don't trigger OCR."""
        # DOCX with little text should NOT trigger OCR
        assert not parser._needs_ocr("test.docx", "")
        assert not parser._needs_ocr("test.docx", "abc")

    @pytest.mark.asyncio
    async def test_ocr_fallback_unavailable(self, parser):
        """Test OCR fallback gracefully handles unavailable service."""
        # Use a non-existent endpoint
        parser.ocr_endpoint = "http://localhost:99999/v1/ocr"

        # Create a temp file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name

        try:
            result = await parser._ocr_fallback(temp_file)
            # Should return None when service unavailable
            assert result is None
        finally:
            os.unlink(temp_file)

    def test_parse_with_metadata(self, parser):
        """Test parsing with metadata."""
        pdf_file = SAMPLE_FILES_DIR / "Alpaca-Care-for-Beginners.pdf"
        assert pdf_file.exists()

        metadata = {
            "filename": "Alpaca-Care-for-Beginners.pdf",
            "content_type": "application/pdf"
        }

        result = parser.parse_sync(str(pdf_file), metadata)

        assert result is not None
        assert len(result) > 0

    def test_parse_multiple_html_files(self, parser):
        """Test parsing multiple HTML files."""
        html_files = [
            "ai_breakthrough.html",
            "quantum_computing_milestone.html",
            "climate_tech_report.html",
        ]

        for html_file in html_files:
            file_path = SAMPLE_FILES_DIR / "news_articles" / html_file
            if not file_path.exists():
                pytest.skip(f"Test file not found: {file_path}")

            result = parser.parse_sync(str(file_path))

            assert result is not None
            assert len(result) > 0, f"Failed to parse {html_file}"

    def test_parse_nonexistent_file(self, parser):
        """Test parsing raises error for nonexistent file."""
        with pytest.raises(Exception):
            parser.parse_sync("/nonexistent/file.pdf")

    @pytest.mark.skipif(
        not os.path.exists("/Users/robthelen/llamafarm-head/llamafarm/examples"),
        reason="Examples directory not found"
    )
    def test_examples_directory_exists(self):
        """Verify examples directory exists."""
        assert EXAMPLES_DIR.exists()
        assert SAMPLE_FILES_DIR.exists()
        assert (SAMPLE_FILES_DIR / "Alpaca-Care-for-Beginners.pdf").exists()
