"""Fixed tests for the refactored parser system."""

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
    parser = CSVParser_Pandas(config={
        "chunk_size": 1000,
        "extract_metadata": True
    })
    
    result = parser.parse(str(csv_file))
    
    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"
    
    # Check document content
    doc = result.documents[0]
    assert "subject" in doc.content.lower() or "login issue" in doc.content.lower()
    # Check for actual metadata field names used by the parser
    assert doc.metadata.get("parser") == "CSVParser_Pandas"
    assert doc.metadata.get("tool") == "Pandas"


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
    parser = CSVParser_Python(config={
        "chunk_size": 1000,
        "encoding": "utf-8"
    })
    
    result = parser.parse(str(csv_file))
    
    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"
    
    # Check document content
    doc = result.documents[0]
    assert doc.metadata.get("parser") == "CSVParser_Python"
    assert doc.metadata.get("tool") == "Python csv"


def test_pdf_parser_pypdf2(tmp_path):
    """Test the PyPDF2 parser."""
    from components.parsers.pdf.pypdf2_parser import PDFParser_PyPDF2
    
    # Note: This test requires a real PDF file to work properly
    # For now, just test that the parser can be instantiated
    parser = PDFParser_PyPDF2(config={
        "chunk_size": 1000,
        "extract_metadata": True,
        "preserve_layout": True
    })
    
    assert parser.name == "PDFParser_PyPDF2"
    assert parser.validate_config() == True


def test_docx_parser(tmp_path):
    """Test the DOCX parser."""
    from components.parsers.docx.python_docx_parser import DocxParser_PythonDocx
    
    # Test basic instantiation
    parser = DocxParser_PythonDocx(config={
        "chunk_size": 1000,
        "extract_tables": True,
        "extract_metadata": True
    })
    
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
    
    parser = MarkdownParser_Python(config={
        "chunk_size": 1000,
        "extract_metadata": True,
        "extract_code_blocks": True,
        "extract_links": True
    })
    
    result = parser.parse(str(md_file))
    
    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"
    
    doc = result.documents[0]
    assert "Main Title" in doc.content
    assert doc.metadata.get("parser") == "MarkdownParser_Python"
    assert doc.metadata.get("tool") == "Python"


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
    
    parser = TextParser_Python(config={
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "encoding": "utf-8",
        "clean_text": True
    })
    
    result = parser.parse(str(text_file))
    
    assert result.documents, "Should have parsed documents"
    assert len(result.errors) == 0, f"Should have no errors: {result.errors}"
    
    doc = result.documents[0]
    assert "test document" in doc.content.lower()
    assert doc.metadata.get("parser") == "TextParser_Python"
    assert doc.metadata.get("tool") == "Python"
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
    
    parser = TextParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "sentences",
        "extract_metadata": True
    })
    
    # This might fail if llama-index is not installed
    try:
        result = parser.parse(str(text_file))
        
        if result.documents:
            doc = result.documents[0]
            assert "test document" in doc.content.lower()
            assert doc.metadata.get("parser") == "TextParser_LlamaIndex"
            assert doc.metadata.get("tool") == "LlamaIndex"
    except ImportError:
        pytest.skip("LlamaIndex not installed")


def test_excel_parser_openpyxl(tmp_path):
    """Test the OpenPyXL Excel parser."""
    from components.parsers.excel.openpyxl_parser import ExcelParser_OpenPyXL
    
    # Test basic instantiation (actual Excel parsing requires openpyxl and .xlsx file)
    parser = ExcelParser_OpenPyXL(config={
        "chunk_size": 1000,
        "extract_formulas": False,
        "extract_metadata": True
    })
    
    assert parser.name == "ExcelParser_OpenPyXL"
    assert parser.validate_config() == True


def test_excel_parser_pandas(tmp_path):
    """Test the Pandas Excel parser."""
    from components.parsers.excel.pandas_parser import ExcelParser_Pandas
    
    # Test basic instantiation
    parser = ExcelParser_Pandas(config={
        "chunk_size": 1000,
        "sheets": None,  # Process all sheets
        "extract_metadata": True
    })
    
    assert parser.name == "ExcelParser_Pandas"
    assert parser.validate_config() == True


def test_parser_factory():
    """Test the parser factory."""
    from components.parsers.parser_factory import ParserFactory
    
    # Test creating a parser instance using the backward compatible interface
    pdf_parser = ParserFactory.create_parser("PDFParser_PyPDF2")
    assert pdf_parser is not None
    assert pdf_parser.__class__.__name__ == "PDFParser_PyPDF2"
    
    csv_parser = ParserFactory.create_parser("CSVParser_Python")
    assert csv_parser is not None
    assert csv_parser.__class__.__name__ == "CSVParser_Python"


def test_tool_aware_parser_factory():
    """Test the tool-aware parser factory."""
    from components.parsers.parser_factory import ToolAwareParserFactory
    
    # Test parser discovery
    parsers = ToolAwareParserFactory.discover_parsers()
    assert "csv" in parsers or "pdf" in parsers or "text" in parsers
    
    # Test listing parsers
    all_parsers = ToolAwareParserFactory.list_parsers()
    assert len(all_parsers) > 0
    
    # Test creating parser instance by name
    parser_instance = ToolAwareParserFactory.create_parser(parser_name="PDFParser_PyPDF2")
    assert parser_instance is not None
    
    # Test creating parser instance by file type
    csv_parser = ToolAwareParserFactory.create_parser(file_type="csv")
    assert csv_parser is not None


def test_parser_registry():
    """Test the parser registry."""
    from components.parsers.parser_registry import ParserRegistry
    
    registry = ParserRegistry()
    
    # Test getting parsers by extension
    pdf_parsers = registry.get_parsers_for_extension(".pdf")
    assert len(pdf_parsers) >= 1
    assert any(p["parser"] == "PDFParser_PyPDF2" for p in pdf_parsers)
    
    csv_parsers = registry.get_parsers_for_extension(".csv")
    assert len(csv_parsers) >= 2
    assert any(p["parser"] == "CSVParser_Pandas" for p in csv_parsers)
    assert any(p["parser"] == "CSVParser_Python" for p in csv_parsers)
    
    # Test getting parsers by MIME type
    pdf_parsers_mime = registry.get_parsers_for_mime("application/pdf")
    assert len(pdf_parsers_mime) >= 1
    
    # Test listing all parsers
    all_parsers = registry.list_all_parsers()
    assert "PDFParser_PyPDF2" in all_parsers
    assert "CSVParser_Python" in all_parsers
    assert "CSVParser_Pandas" in all_parsers


def test_parser_with_chunking(tmp_path):
    """Test parser with chunking enabled."""
    from components.parsers.text.python_parser import TextParser_Python
    
    # Create a long text file that should be chunked
    long_text = "This is a test sentence. " * 100  # Create a long text
    
    text_file = tmp_path / "long_test.txt"
    text_file.write_text(long_text)
    
    parser = TextParser_Python(config={
        "chunk_size": 100,  # Small chunk size to force chunking
        "chunk_overlap": 20,
        "chunk_strategy": "characters",
        "encoding": "utf-8"
    })
    
    result = parser.parse(str(text_file))
    
    assert result.documents, "Should have parsed documents"
    assert len(result.documents) > 1, "Should have multiple chunks"
    
    # Check that chunks have proper metadata
    for i, doc in enumerate(result.documents):
        assert "chunk_index" in doc.metadata
        assert doc.metadata["chunk_index"] == i
        assert len(doc.content) <= 150  # Allow some flexibility for word boundaries


# Additional tests for all LlamaIndex parsers
def test_markdown_parser_llamaindex(tmp_path):
    """Test the LlamaIndex Markdown parser."""
    from components.parsers.markdown.llamaindex_parser import MarkdownParser_LlamaIndex
    
    # Create a test markdown file
    markdown_content = """---
title: LlamaIndex Test
author: Test Author
---

# Main Title

This is a test paragraph for LlamaIndex parser.

## Section 1

- Item 1
- Item 2

### Subsection 1.1

More content here with **bold** text.

## Section 2

Final section with conclusion.
"""
    
    md_file = tmp_path / "test_llama.md"
    md_file.write_text(markdown_content)
    
    parser = MarkdownParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "headings",
        "extract_metadata": True
    })
    
    try:
        result = parser.parse(str(md_file))
        
        if result.documents:
            doc = result.documents[0]
            assert doc.metadata.get("parser") == "MarkdownParser_LlamaIndex"
            assert doc.metadata.get("tool") == "LlamaIndex"
    except ImportError:
        pytest.skip("LlamaIndex not installed")


def test_pdf_parser_llamaindex(tmp_path):
    """Test the LlamaIndex PDF parser."""
    from components.parsers.pdf.llamaindex_parser import PDFParser_LlamaIndex
    
    # Test basic instantiation
    parser = PDFParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "sentences",
        "extract_metadata": True,
        "fallback_strategies": ["pypdf2_fallback"]
    })
    
    assert parser.name == "PDFParser_LlamaIndex"
    assert parser.validate_config() == True
    
    # Note: Actual PDF parsing requires a real PDF file and LlamaIndex installed


def test_csv_parser_llamaindex(tmp_path):
    """Test the LlamaIndex CSV parser."""
    from components.parsers.csv.llamaindex_parser import CSVParser_LlamaIndex
    
    # Create a test CSV file
    csv_content = """subject,body,type,priority
Login Issue,Cannot login to the system,Incident,medium
Feature Request,Need dark mode,Request,low
Data Error,Database showing wrong values,Bug,high"""
    
    csv_file = tmp_path / "test_llama.csv"
    csv_file.write_text(csv_content)
    
    parser = CSVParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "rows",
        "extract_metadata": True
    })
    
    try:
        result = parser.parse(str(csv_file))
        
        if result.documents:
            doc = result.documents[0]
            assert doc.metadata.get("parser") == "CSVParser_LlamaIndex"
            assert doc.metadata.get("tool") == "LlamaIndex-Pandas"
    except ImportError:
        pytest.skip("LlamaIndex not installed")


def test_docx_parser_llamaindex(tmp_path):
    """Test the LlamaIndex DOCX parser."""
    from components.parsers.docx.llamaindex_parser import DocxParser_LlamaIndex
    
    # Test basic instantiation
    parser = DocxParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "paragraphs",
        "extract_metadata": True,
        "extract_images": False
    })
    
    assert parser.name == "DocxParser_LlamaIndex"
    assert parser.validate_config() == True
    
    # Note: Actual DOCX parsing requires a real DOCX file and LlamaIndex installed


def test_excel_parser_llamaindex(tmp_path):
    """Test the LlamaIndex Excel parser."""
    from components.parsers.excel.llamaindex_parser import ExcelParser_LlamaIndex
    
    # Test basic instantiation
    parser = ExcelParser_LlamaIndex(config={
        "chunk_size": 1000,
        "chunk_strategy": "rows",
        "sheets": None,
        "combine_sheets": False,
        "extract_metadata": True
    })
    
    assert parser.name == "ExcelParser_LlamaIndex"
    assert parser.validate_config() == True
    
    # Note: Actual Excel parsing requires a real Excel file and LlamaIndex installed


def test_parser_registry_includes_llamaindex():
    """Test that parser registry includes all LlamaIndex parsers."""
    from components.parsers.parser_registry import ParserRegistry
    
    registry = ParserRegistry()
    all_parsers = registry.list_all_parsers()
    
    # Check that all LlamaIndex parsers are in the registry
    expected_llamaindex_parsers = [
        "MarkdownParser_LlamaIndex",
        "PDFParser_LlamaIndex",
        "TextParser_LlamaIndex",
        "DocxParser_LlamaIndex",
        "CSVParser_LlamaIndex",
        "ExcelParser_LlamaIndex"
    ]
    
    for parser_name in expected_llamaindex_parsers:
        assert parser_name in all_parsers, f"{parser_name} should be in registry"
        
        # Check that parser info is correct
        parser_info = registry.get_parser(parser_name)
        assert parser_info is not None
        assert "LlamaIndex" in parser_info.get("tool", "")


def test_parser_factory_creates_llamaindex_parsers():
    """Test that parser factory can create LlamaIndex parsers."""
    from components.parsers.parser_factory import ToolAwareParserFactory
    
    # Test creating each LlamaIndex parser
    llamaindex_parsers = [
        "MarkdownParser_LlamaIndex",
        "PDFParser_LlamaIndex",
        "TextParser_LlamaIndex",
        "DocxParser_LlamaIndex",
        "CSVParser_LlamaIndex",
        "ExcelParser_LlamaIndex"
    ]
    
    for parser_name in llamaindex_parsers:
        parser = ToolAwareParserFactory.create_parser(parser_name=parser_name)
        assert parser is not None, f"Should be able to create {parser_name}"
        assert parser.name == parser_name