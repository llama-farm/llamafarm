"""Updated tests for the refactored parser system."""

import pytest
from pathlib import Path
import tempfile
import os


# Test the CSV parsers
def test_csv_parser_pandas(tmp_path):
    """Test the Pandas CSV parser."""
    from components.parsers.csv.pandas_parser import CSVParser_Pandas

    # Create a test CSV file
    csv_content = """subject,body,type,priority
Login Issue,Cannot login to the system,Incident,medium
Feature Request,Need dark mode,Request,low
Data Error,Database showing wrong values,Bug,high"""

    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Test basic parsing
    parser = CSVParser_Pandas(config={"chunk_size": 1000, "extract_metadata": True})

    result = parser.parse(str(csv_file))

    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"

    # Check document content
    doc = result.documents[0]
    assert "subject" in doc.content.lower() or "login issue" in doc.content.lower()
    # Accept current metadata keys
    assert (
        doc.metadata.get("parser") == "CSVParser_Pandas"
        or doc.metadata.get("parser_type") == "CSVParser_Pandas"
    )


def test_csv_parser_python(tmp_path):
    """Test the Python CSV parser."""
    from components.parsers.csv.python_parser import CSVParser_Python

    # Create a test CSV file
    csv_content = """subject,body,type,priority
Login Issue,Cannot login to the system,Incident,medium
Feature Request,Need dark mode,Request,low
Data Error,Database showing wrong values,Bug,high"""

    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)

    # Test basic parsing
    parser = CSVParser_Python(config={"chunk_size": 1000, "encoding": "utf-8"})

    result = parser.parse(str(csv_file))

    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"

    # Check document content
    doc = result.documents[0]
    assert (
        doc.metadata.get("parser") == "CSVParser_Python"
        or doc.metadata.get("parser_type") == "CSVParser_Python"
    )


def test_pdf_parser_pypdf2(tmp_path):
    """Test the PyPDF2 parser."""
    from components.parsers.pdf.pypdf2_parser import PDFParser_PyPDF2

    # Note: This test requires a real PDF file to work properly
    # For now, just test that the parser can be instantiated
    parser = PDFParser_PyPDF2(
        config={"chunk_size": 1000, "extract_metadata": True, "preserve_layout": True}
    )

    assert parser.name == "PDFParser_PyPDF2"
    assert parser.validate_config() == True


def test_docx_parser(tmp_path):
    """Test the DOCX parser."""
    from components.parsers.docx.python_docx_parser import DocxParser_PythonDocx

    # Test basic instantiation
    parser = DocxParser_PythonDocx(
        config={"chunk_size": 1000, "extract_tables": True, "extract_metadata": True}
    )

    assert parser.name == "DocxParser_PythonDocx"
    assert parser.validate_config() == True


def test_markdown_parser(tmp_path):
    """Test the Markdown parser."""
    from components.parsers.markdown.python_parser import MarkdownParser_Python

    # Create a test markdown file
    markdown_content = """---
title: Test Document
author: Test Author
---

# Main Title

This is a test paragraph with **bold** and *italic* text.

## Section 1

- Item 1
- Item 2
- Item 3

```python
def hello():
    print("Hello World")
```

[Link to example](https://example.com)
"""

    md_file = tmp_path / "test.md"
    md_file.write_text(markdown_content)

    parser = MarkdownParser_Python(
        config={
            "chunk_size": 1000,
            "extract_metadata": True,
            "extract_code_blocks": True,
            "extract_links": True,
        }
    )

    result = parser.parse(str(md_file))

    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"

    doc = result.documents[0]
    assert "Main Title" in doc.content
    assert (
        doc.metadata.get("parser") == "MarkdownParser_Python"
        or doc.metadata.get("parser_type") == "MarkdownParser_Python"
    )


def test_text_parser_python(tmp_path):
    """Test the Python text parser."""
    from components.parsers.text.python_parser import TextParser_Python

    # Create a test text file
    text_content = """This is a test document.

It has multiple paragraphs.

And some structured content:
- Item 1
- Item 2
- Item 3

Final paragraph with conclusion."""

    text_file = tmp_path / "test.txt"
    text_file.write_text(text_content)

    parser = TextParser_Python(
        config={
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "encoding": "utf-8",
            "clean_text": True,
        }
    )

    result = parser.parse(str(text_file))

    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"

    doc = result.documents[0]
    assert "test document" in doc.content.lower()
    assert (
        doc.metadata.get("parser") == "TextParser_Python"
        or doc.metadata.get("parser_type") == "TextParser_Python"
    )
    assert "word_count" in doc.metadata


def test_text_parser_llamaindex(tmp_path):
    """Test the LlamaIndex text parser."""
    from components.parsers.text.llamaindex_parser import TextParser_LlamaIndex

    # Create a test text file
    text_content = """This is a test document for LlamaIndex parser.

It supports advanced chunking strategies including semantic splitting.

```python
# Code blocks are preserved
def example():
    return "test"
```

The parser can handle various text formats."""

    text_file = tmp_path / "test.txt"
    text_file.write_text(text_content)

    parser = TextParser_LlamaIndex(
        config={
            "chunk_size": 1000,
            "chunk_strategy": "sentences",
            "extract_metadata": True,
        }
    )

    # This might fail if llama-index is not installed
    try:
        result = parser.parse(str(text_file))

        if result.documents:
            doc = result.documents[0]
            assert "test document" in doc.content.lower()
            assert doc.metadata.get("parser_type") == "TextParser_LlamaIndex"
    except ImportError:
        pytest.skip("LlamaIndex not installed")


def test_excel_parser_openpyxl(tmp_path):
    """Test the OpenPyXL Excel parser."""
    from components.parsers.excel.openpyxl_parser import ExcelParser_OpenPyXL

    # Test basic instantiation (actual Excel parsing requires openpyxl and .xlsx file)
    parser = ExcelParser_OpenPyXL(
        config={"chunk_size": 1000, "extract_formulas": False, "extract_metadata": True}
    )

    assert parser.name == "ExcelParser_OpenPyXL"
    assert parser.validate_config() == True


def test_excel_parser_pandas(tmp_path):
    """Test the Pandas Excel parser."""
    from components.parsers.excel.pandas_parser import ExcelParser_Pandas

    # Test basic instantiation
    parser = ExcelParser_Pandas(
        config={
            "chunk_size": 1000,
            "sheets": None,  # Process all sheets
            "extract_metadata": True,
        }
    )

    assert parser.name == "ExcelParser_Pandas"
    assert parser.validate_config() == True


def test_parser_factory():
    """Test the parser factory (current API)."""
    from components.parsers.parser_factory import ParserFactory

    # Create parser instances via backward-compatible interface
    pdf_parser = ParserFactory.create_parser("PDFParser_PyPDF2")
    assert pdf_parser is not None
    assert pdf_parser.__class__.__name__ == "PDFParser_PyPDF2"

    csv_parser = ParserFactory.create_parser("CSVParser_Python")
    assert csv_parser is not None
    assert csv_parser.__class__.__name__ == "CSVParser_Python"


def test_parser_registry():
    """Test the parser registry (aligned to current registry API)."""
    from components.parsers.parser_registry import ParserRegistry

    registry = ParserRegistry()

    # Get parsers by file extension
    pdf_parsers = registry.get_parsers_for_extension(".pdf")
    assert len(pdf_parsers) >= 1
    assert any(p.get("parser") == "PDFParser_PyPDF2" for p in pdf_parsers)

    csv_parsers = registry.get_parsers_for_extension(".csv")
    assert len(csv_parsers) >= 2
    assert any(p.get("parser") == "CSVParser_Pandas" for p in csv_parsers)

    # Basic listing
    all_parsers = registry.list_all_parsers()
    assert "PDFParser_PyPDF2" in all_parsers
    assert "CSVParser_Python" in all_parsers
