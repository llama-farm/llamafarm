"""
Tests for the MarkItDownParser.

Tests cover:
- Basic imports and initialization
- Document conversion to markdown
- Metadata extraction and preservation
- Content statistics extraction
- Error handling
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if markitdown is available
MARKITDOWN_AVAILABLE = False
try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    pass


# Skip all tests if markitdown is not installed
pytestmark = pytest.mark.skipif(
    not MARKITDOWN_AVAILABLE,
    reason="MarkItDown not installed. Install with: pip install 'markitdown[all]'"
)


class TestMarkItDownParserImports:
    """Test that MarkItDownParser can be imported."""

    def test_markitdown_parser_import(self):
        """Test that MarkItDownParser can be imported."""
        from components.parsers.markitdown import MarkItDownParser
        assert MarkItDownParser is not None

    def test_markitdown_parser_initialization(self):
        """Test basic parser initialization."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()
        assert parser.name == "MarkItDownParser"
        assert parser.output_format == "markdown"  # Default
        assert parser.extract_metadata is True  # Default

    def test_markitdown_parser_custom_config(self):
        """Test parser initialization with custom config."""
        from components.parsers.markitdown import MarkItDownParser

        config = {
            "output_format": "text",
            "extract_metadata": False,
            "use_llm_for_images": False,
        }
        parser = MarkItDownParser(config=config)

        assert parser.output_format == "text"
        assert parser.extract_metadata is False


class TestMarkItDownParserCanParse:
    """Test the can_parse method."""

    def test_can_parse_pdf(self):
        """Test that parser can handle PDF files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("document.pdf") is True
        assert parser.can_parse("report.PDF") is True

    def test_can_parse_docx(self):
        """Test that parser can handle DOCX files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("document.docx") is True

    def test_can_parse_xlsx(self):
        """Test that parser can handle Excel files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("data.xlsx") is True
        assert parser.can_parse("data.xls") is True

    def test_can_parse_html(self):
        """Test that parser can handle HTML files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("page.html") is True
        assert parser.can_parse("page.htm") is True

    def test_can_parse_csv(self):
        """Test that parser can handle CSV files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("data.csv") is True

    def test_can_parse_json(self):
        """Test that parser can handle JSON files."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("data.json") is True

    def test_cannot_parse_unsupported(self):
        """Test that parser rejects unsupported formats."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        assert parser.can_parse("video.mp4") is False
        assert parser.can_parse("archive.zip") is False


class TestMarkItDownParserMetadata:
    """Test metadata extraction capabilities."""

    def test_build_base_metadata(self):
        """Test base metadata building from input metadata."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        input_metadata = {
            "dataset_name": "test_dataset",
            "upload_timestamp": "2024-01-15T10:30:00",
            "file_hash": "abc123",
            "content_type": "application/pdf",
            "namespace": "test_ns",
            "project": "test_project",
            "extra_field": "should_not_be_included",
        }

        base_metadata = parser._build_base_metadata(input_metadata)

        # Check preserved fields
        assert base_metadata.get("dataset_name") == "test_dataset"
        assert base_metadata.get("upload_timestamp") == "2024-01-15T10:30:00"
        assert base_metadata.get("file_hash") == "abc123"
        assert base_metadata.get("namespace") == "test_ns"
        assert base_metadata.get("project") == "test_project"

        # Extra field should not be included
        assert "extra_field" not in base_metadata

    def test_generate_doc_id(self):
        """Test document ID generation."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        content = "This is test content for hashing"
        metadata = {"source_filename": "test_document.pdf"}

        # Without chunk index
        doc_id = parser._generate_doc_id(content, metadata)
        assert "test_document_" in doc_id
        assert len(doc_id) > 10

        # With chunk index
        doc_id_chunk = parser._generate_doc_id(content, metadata, chunk_index=0)
        assert "_c0" in doc_id_chunk

    def test_load_metadata(self):
        """Test parser metadata loading."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        parser_config = parser._load_metadata()

        assert parser_config.name == "MarkItDownParser"
        assert ".pdf" in parser_config.supported_extensions
        assert "text_extraction" in parser_config.capabilities
        assert "markitdown" in parser_config.dependencies["required"]


class TestMarkItDownParserContentStats:
    """Test content statistics extraction."""

    def test_extract_content_stats_basic(self):
        """Test basic content statistics extraction."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        content = """# Heading 1

This is a paragraph with some text.

## Heading 2

Another paragraph here.

- List item 1
- List item 2
"""
        stats = parser._extract_content_stats(content)

        assert stats["char_count"] == len(content)
        assert stats["word_count"] > 0
        assert stats["line_count"] > 0
        assert stats.get("heading_count", 0) == 2  # Two headings
        assert stats.get("list_item_count", 0) >= 2  # Two list items

    def test_extract_content_stats_with_table(self):
        """Test content statistics with table detection."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        content = """# Data Table

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |
"""
        stats = parser._extract_content_stats(content)

        assert stats.get("has_tables") is True
        assert stats.get("table_line_count", 0) >= 3

    def test_extract_content_stats_with_code(self):
        """Test content statistics with code blocks."""
        from components.parsers.markitdown import MarkItDownParser
        parser = MarkItDownParser()

        content = """# Code Example

```python
def hello():
    print("Hello, World!")
```

Some text after code.

```javascript
console.log("test");
```
"""
        stats = parser._extract_content_stats(content)

        assert stats.get("code_block_count", 0) == 2


class TestMarkItDownParserBlobParsing:
    """Test blob parsing with metadata preservation."""

    def test_parse_blob_preserves_metadata(self):
        """Test that parse_blob preserves input metadata."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        # Create simple test content
        test_content = b"Hello World - this is a test document."

        input_metadata = {
            "filename": "test.txt",
            "dataset_name": "test_dataset",
            "upload_timestamp": "2024-01-15T10:30:00",
            "namespace": "test_ns",
            "project": "test_project",
        }

        # Mock the converter to return predictable results
        with patch.object(parser, 'converter') as mock_converter:
            mock_result = MagicMock()
            mock_result.text_content = "Hello World - this is a test document."
            mock_converter.convert.return_value = mock_result

            documents = parser.parse_blob(test_content, input_metadata)

        assert len(documents) >= 1
        doc = documents[0]

        # Check metadata is preserved
        assert doc.metadata.get("dataset_name") == "test_dataset"
        assert doc.metadata.get("namespace") == "test_ns"
        assert doc.metadata.get("project") == "test_project"
        assert doc.metadata.get("source_filename") == "test.txt"
        assert doc.metadata.get("parser_type") == "MarkItDownParser"

    def test_parse_blob_includes_content_stats(self):
        """Test that parse_blob includes content statistics."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        # Create test content with markdown
        test_content = b"# Title\n\nSome content here."

        input_metadata = {"filename": "test.md"}

        with patch.object(parser, 'converter') as mock_converter:
            mock_result = MagicMock()
            mock_result.text_content = "# Title\n\nSome content here."
            mock_converter.convert.return_value = mock_result

            documents = parser.parse_blob(test_content, input_metadata)

        assert len(documents) >= 1
        doc = documents[0]

        # Check content stats are included
        assert doc.metadata.get("char_count") > 0
        assert doc.metadata.get("word_count") > 0
        assert doc.metadata.get("line_count") > 0


@pytest.mark.integration
class TestMarkItDownParserIntegration:
    """Integration tests that require real MarkItDown processing."""

    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing."""
        content = """# Test Document

This is a test document with some content.

## Section 1

Some text in section 1.

- Item 1
- Item 2

## Section 2

More content here.
"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write(content)
            return f.name

    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing."""
        content = """Name,Age,City
John,30,New York
Jane,25,Los Angeles
Bob,35,Chicago
"""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write(content)
            return f.name

    def test_parse_text_file(self, sample_text_file):
        """Test parsing a text file."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        try:
            result = parser.parse(sample_text_file)

            assert result.documents is not None
            assert len(result.documents) > 0
            assert len(result.errors) == 0

            doc = result.documents[0]
            assert doc.content is not None
            assert len(doc.content) > 0
            assert doc.metadata.get("parser_type") == "MarkItDownParser"
        finally:
            os.unlink(sample_text_file)

    def test_parse_csv_file(self, sample_csv_file):
        """Test parsing a CSV file."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        try:
            result = parser.parse(sample_csv_file)

            assert result.documents is not None
            assert len(result.documents) > 0
            assert len(result.errors) == 0

            doc = result.documents[0]
            assert doc.content is not None
            # CSV should be converted to markdown table format
            assert "|" in doc.content  # Markdown table indicator
        finally:
            os.unlink(sample_csv_file)


class TestMarkItDownParserErrorHandling:
    """Test error handling capabilities."""

    def test_parse_nonexistent_file(self):
        """Test handling of nonexistent files."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()
        result = parser.parse("/nonexistent/path/document.pdf")

        assert len(result.errors) > 0
        assert "not found" in str(result.errors[0].get("error", "")).lower() or \
               "no such file" in str(result.errors[0].get("error", "")).lower()

    def test_parse_blob_returns_error_document_on_failure(self):
        """Test that parse_blob returns a document with error info on failure."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        # Mock converter to raise an exception
        with patch.object(parser, 'converter') as mock_converter:
            mock_converter.convert.side_effect = Exception("Conversion failed")

            metadata = {"filename": "test.pdf"}
            documents = parser.parse_blob(b"invalid data", metadata)

            assert len(documents) >= 1
            doc = documents[0]
            assert doc.metadata.get("parsing_error") is not None
            assert "Conversion failed" in doc.metadata.get("parsing_error", "")


class TestMarkItDownParserNoChunking:
    """Test that MarkItDown doesn't chunk by default."""

    def test_documents_not_chunked_by_default(self):
        """Test that documents are not chunked by default."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser()

        with patch.object(parser, 'converter') as mock_converter:
            mock_result = MagicMock()
            mock_result.text_content = "Test content"
            mock_converter.convert.return_value = mock_result

            documents = parser.parse_blob(b"test", {"filename": "test.txt"})

            assert len(documents) == 1
            doc = documents[0]
            assert doc.metadata.get("is_chunked") is False
            assert doc.metadata.get("total_chunks") == 1
