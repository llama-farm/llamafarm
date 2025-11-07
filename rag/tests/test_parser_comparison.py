"""Comprehensive parser comparison tests using real sample files."""

import pytest
from pathlib import Path
from components.parsers.parser_factory import ToolAwareParserFactory
from components.parsers.markitdown.markitdown_parser import MarkItDownParser


# Sample files directory
SAMPLES_DIR = Path(__file__).parent.parent.parent / ".plans" / "samples"

# Test fixtures for sample files
PDF_FILE = SAMPLES_DIR / "llamafarm - Healthcare - Aug 2025 2 .pdf"
DOCX_FILE = SAMPLES_DIR / "Why Decentralized AI Is Inevitable_ A Developer's Manifesto.docx"
XLSX_FILE = SAMPLES_DIR / "rownd_vs_firebase_comparison.xlsx"
HTML_FILE = SAMPLES_DIR / "Daily Diet, Treats, And Supplements For Llamas - The Open Sanctuary Project.html"
PPTX_FILE = SAMPLES_DIR / "LlamaFarm.pptx"
PNG_FILE = SAMPLES_DIR / "ChatGPT Image Sep 29, 2025, 02_21_26 PM.png"


class TestMarkItDownParser:
    """Test MarkItDown universal parser."""

    def test_markitdown_pdf_parsing(self):
        """Test MarkItDown can parse PDF files."""
        if not PDF_FILE.exists():
            pytest.skip(f"Sample file not found: {PDF_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(PDF_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 0, "Should extract content"
        assert result.documents[0].metadata["parser"] == "MarkItDownParser"
        assert result.documents[0].metadata["original_format"] == ".pdf"

    def test_markitdown_docx_parsing(self):
        """Test MarkItDown can parse DOCX files."""
        if not DOCX_FILE.exists():
            pytest.skip(f"Sample file not found: {DOCX_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(DOCX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 20000, "Should extract substantial content"
        assert "DECENTRALIZED AI" in result.documents[0].content.upper()

    def test_markitdown_xlsx_parsing(self):
        """Test MarkItDown can parse XLSX files."""
        if not XLSX_FILE.exists():
            pytest.skip(f"Sample file not found: {XLSX_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(XLSX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert "|" in result.documents[0].content, "Should create markdown tables"
        assert "Rownd" in result.documents[0].content
        assert "Firebase" in result.documents[0].content

    def test_markitdown_html_parsing(self):
        """Test MarkItDown can parse HTML files."""
        if not HTML_FILE.exists():
            pytest.skip(f"Sample file not found: {HTML_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(HTML_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 60000, "Should extract substantial content"
        assert "llama" in result.documents[0].content.lower()

    def test_markitdown_pptx_parsing(self):
        """Test MarkItDown can parse PPTX files."""
        if not PPTX_FILE.exists():
            pytest.skip(f"Sample file not found: {PPTX_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(PPTX_FILE))

        assert len(result.documents) > 0, "Should extract at least one document"
        assert len(result.documents[0].content) > 4000, "Should extract substantial content"
        assert "LlamaFarm" in result.documents[0].content

    def test_markitdown_png_metadata(self):
        """Test MarkItDown extracts PNG metadata."""
        if not PNG_FILE.exists():
            pytest.skip(f"Sample file not found: {PNG_FILE}")

        parser = MarkItDownParser(config={"chain_to_markdown_parser": False})
        result = parser.parse(str(PNG_FILE))

        assert len(result.documents) > 0, "Should extract at least metadata"
        assert result.documents[0].metadata["original_format"] == ".png"

    def test_markitdown_chained_mode(self):
        """Test MarkItDown with chained markdown parser."""
        if not HTML_FILE.exists():
            pytest.skip(f"Sample file not found: {HTML_FILE}")

        config = {
            "chain_to_markdown_parser": True,
            "markdown_parser": "MarkdownParser_Python",
            "chunk_size": 1000,
            "chunk_strategy": "sections",
            "chunk_overlap": 100,
        }
        parser = MarkItDownParser(config=config)
        result = parser.parse(str(HTML_FILE))

        assert len(result.documents) > 1, "Should create multiple chunks"
        assert result.documents[0].metadata["preprocessing"] == "markitdown"
        assert result.documents[0].metadata["chunking_parser"] == "MarkdownParser_Python"
        assert "chunk_index" in result.documents[0].metadata
        assert "total_chunks" in result.documents[0].metadata


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


class TestParserComparison:
    """Compare MarkItDown vs specialized parsers."""

    def test_pdf_content_comparison(self):
        """Compare PDF content extraction between parsers."""
        if not PDF_FILE.exists():
            pytest.skip(f"Sample file not found: {PDF_FILE}")

        markitdown = MarkItDownParser(config={"chain_to_markdown_parser": False})
        pypdf2 = ToolAwareParserFactory.create_parser("PDFParser_PyPDF2", config={})

        md_result = markitdown.parse(str(PDF_FILE))
        pypdf2_result = pypdf2.parse(str(PDF_FILE))

        assert len(md_result.documents) > 0, "MarkItDown should extract content"
        assert len(pypdf2_result.documents) > 0, "PyPDF2 should extract content"

        # Both should extract meaningful content
        assert len(md_result.documents[0].content) > 2000
        assert len(pypdf2_result.documents[0].content) > 2000

    def test_docx_content_comparison(self):
        """Compare DOCX content extraction between parsers."""
        if not DOCX_FILE.exists():
            pytest.skip(f"Sample file not found: {DOCX_FILE}")

        markitdown = MarkItDownParser(config={"chain_to_markdown_parser": False})
        pythondocx = ToolAwareParserFactory.create_parser("DocxParser_PythonDocx", config={})

        md_result = markitdown.parse(str(DOCX_FILE))
        docx_result = pythondocx.parse(str(DOCX_FILE))

        assert len(md_result.documents) > 0, "MarkItDown should extract content"
        assert len(docx_result.documents) > 0, "PythonDocx should extract content"

        # Both should extract meaningful content (check total across all chunks)
        md_total = sum(len(doc.content) for doc in md_result.documents)
        docx_total = sum(len(doc.content) for doc in docx_result.documents)
        assert md_total > 5000, f"MarkItDown extracted {md_total} chars"
        assert docx_total > 5000, f"PythonDocx extracted {docx_total} chars"

    def test_xlsx_content_comparison(self):
        """Compare XLSX content extraction between parsers."""
        if not XLSX_FILE.exists():
            pytest.skip(f"Sample file not found: {XLSX_FILE}")

        markitdown = MarkItDownParser(config={"chain_to_markdown_parser": False})
        pandas = ToolAwareParserFactory.create_parser("ExcelParser_Pandas", config={})
        llamaindex = ToolAwareParserFactory.create_parser("ExcelParser_LlamaIndex", config={})

        md_result = markitdown.parse(str(XLSX_FILE))
        pandas_result = pandas.parse(str(XLSX_FILE))
        llamaindex_result = llamaindex.parse(str(XLSX_FILE))

        assert len(md_result.documents) > 0, "MarkItDown should extract content"
        assert len(pandas_result.documents) > 0, "Pandas should extract content"
        assert len(llamaindex_result.documents) > 0, "LlamaIndex should extract content"

        # All parsers should extract meaningful content from Excel
        assert len(md_result.documents[0].content) > 500
        assert len(pandas_result.documents[0].content) > 500
        assert len(llamaindex_result.documents[0].content) > 500


class TestParserRegistry:
    """Test parser discovery and registration."""

    def test_markitdown_parser_discoverable(self):
        """Test MarkItDownParser is discoverable."""
        parsers = ToolAwareParserFactory.list_parsers()
        assert "MarkItDownParser" in parsers, "MarkItDownParser should be in registry"

    def test_markitdown_parser_info(self):
        """Test MarkItDownParser metadata."""
        info = ToolAwareParserFactory.get_parser_info("MarkItDownParser")
        assert info is not None, "Should find parser info"
        assert info["name"] == "MarkItDownParser"
        assert info["tool"] == "MarkItDown"
        assert ".pdf" in info["supported_extensions"]
        assert ".docx" in info["supported_extensions"]
        assert ".pptx" in info["supported_extensions"]
        assert ".xlsx" in info["supported_extensions"]
        assert ".html" in info["supported_extensions"]

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
            "MarkItDownParser",
        ]

        for parser_name in expected_parsers:
            assert parser_name in parsers, f"{parser_name} should be registered"
