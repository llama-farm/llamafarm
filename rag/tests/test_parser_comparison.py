"""Comprehensive parser comparison tests using real sample files."""

import pytest
from pathlib import Path
from components.parsers.parser_factory import ToolAwareParserFactory
from core.base import ProcessingResult


# Sample files directory
SAMPLES_DIR = Path(__file__).parent.parent.parent / ".plans" / "samples"

# Test fixtures for sample files
PDF_FILE = SAMPLES_DIR / "llamafarm - Healthcare - Aug 2025 2 .pdf"
DOCX_FILE = SAMPLES_DIR / "Why Decentralized AI Is Inevitable_ A Developer's Manifesto.docx"
XLSX_FILE = SAMPLES_DIR / "rownd_vs_firebase_comparison.xlsx"
HTML_FILE = SAMPLES_DIR / "Daily Diet, Treats, And Supplements For Llamas - The Open Sanctuary Project.html"
PPTX_FILE = SAMPLES_DIR / "LlamaFarm.pptx"
PNG_FILE = SAMPLES_DIR / "ChatGPT Image Sep 29, 2025, 02_21_26 PM.png"


class TestPDFParsers:
    """Test specialized PDF parsers."""

    def test_pdf_pypdf2_parser(self):
        """Test PDFParser_PyPDF2."""
        if not PDF_FILE.exists():
            pytest.skip(f"Sample file not found: {PDF_FILE}")

        parser = ToolAwareParserFactory.create_parser("PDFParser_PyPDF2", config={})
        result = parser.parse(str(PDF_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 5000, "Should extract substantial content"

    def test_pdf_llamaindex_parser(self):
        """Test PDFParser_LlamaIndex."""
        if not PDF_FILE.exists():
            pytest.skip(f"Sample file not found: {PDF_FILE}")

        parser = ToolAwareParserFactory.create_parser("PDFParser_LlamaIndex", config={})
        result = parser.parse(str(PDF_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 2000, "Should extract substantial content"


class TestDOCXParsers:
    """Test specialized DOCX parsers."""

    def test_docx_pythondocx_parser(self):
        """Test DocxParser_PythonDocx."""
        if not DOCX_FILE.exists():
            pytest.skip(f"Sample file not found: {DOCX_FILE}")

        parser = ToolAwareParserFactory.create_parser("DocxParser_PythonDocx", config={})
        result = parser.parse(str(DOCX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        # Check total content across all chunks (parser creates multiple chunks)
        total_content = sum(len(doc.content) for doc in result.documents)
        assert total_content > 5000, "Should extract substantial content across all chunks"

    def test_docx_llamaindex_parser(self):
        """Test DocxParser_LlamaIndex."""
        if not DOCX_FILE.exists():
            pytest.skip(f"Sample file not found: {DOCX_FILE}")

        parser = ToolAwareParserFactory.create_parser("DocxParser_LlamaIndex", config={})
        result = parser.parse(str(DOCX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        # Check total content across all chunks (parser creates multiple chunks)
        total_content = sum(len(doc.content) for doc in result.documents)
        assert total_content > 5000, "Should extract substantial content across all chunks"


class TestExcelParsers:
    """Test specialized Excel parsers."""

    def test_excel_pandas_parser(self):
        """Test ExcelParser_Pandas."""
        if not XLSX_FILE.exists():
            pytest.skip(f"Sample file not found: {XLSX_FILE}")

        parser = ToolAwareParserFactory.create_parser("ExcelParser_Pandas", config={})
        result = parser.parse(str(XLSX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 500, "Should extract substantial content"
        assert "Rownd" in result.documents[0].content

    def test_excel_llamaindex_parser(self):
        """Test ExcelParser_LlamaIndex."""
        if not XLSX_FILE.exists():
            pytest.skip(f"Sample file not found: {XLSX_FILE}")

        parser = ToolAwareParserFactory.create_parser("ExcelParser_LlamaIndex", config={})
        result = parser.parse(str(XLSX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 500, "Should extract substantial content"


class TestParserRegistry:
    """Test parser discovery and registration."""

    def test_all_specialized_parsers_registered(self):
        """Test all specialized parsers are registered."""
        parsers = ToolAwareParserFactory.list_parsers()

        expected_parsers = [
            "PDFParser_PyPDF2",
            "PDFParser_LlamaIndex",
            "DocxParser_PythonDocx",
            "DocxParser_LlamaIndex",
            "ExcelParser_Pandas",
            "ExcelParser_LlamaIndex",
        ]

        for parser_name in expected_parsers:
            assert parser_name in parsers, f"{parser_name} should be registered"
