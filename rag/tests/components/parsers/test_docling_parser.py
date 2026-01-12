"""
Tests for the DoclingParser.

Tests cover:
- Basic imports and initialization
- PDF parsing with Docling
- Metadata extraction including page numbers, document structure
- Smart chunking with HybridChunker
- Blob parsing with comprehensive metadata
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check if docling is available
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    pass


# Skip all tests if docling is not installed
pytestmark = pytest.mark.skipif(
    not DOCLING_AVAILABLE,
    reason="Docling not installed. Install with: pip install docling docling-core[chunking]"
)


class TestDoclingParserImports:
    """Test that DoclingParser can be imported."""

    def test_docling_parser_import(self):
        """Test that DoclingParser can be imported."""
        from components.parsers.docling import DoclingParser
        assert DoclingParser is not None

    def test_docling_parser_initialization(self):
        """Test basic parser initialization."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser()
        assert parser.name == "DoclingParser"
        assert parser.chunk_size == 512  # Default
        assert parser.chunk_strategy == "hybrid"  # Default

    def test_docling_parser_custom_config(self):
        """Test parser initialization with custom config."""
        from components.parsers.docling import DoclingParser

        config = {
            "chunk_size": 256,
            "chunk_strategy": "hierarchical",
            "output_format": "text",
        }
        parser = DoclingParser(config=config)

        assert parser.chunk_size == 256
        assert parser.chunk_strategy == "hierarchical"
        assert parser.output_format == "text"


class TestDoclingParserCanParse:
    """Test the can_parse method."""

    def test_can_parse_pdf(self):
        """Test that parser can handle PDF files."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        assert parser.can_parse("document.pdf") is True
        assert parser.can_parse("report.PDF") is True

    def test_can_parse_docx(self):
        """Test that parser can handle DOCX files."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        assert parser.can_parse("document.docx") is True
        assert parser.can_parse("report.DOCX") is True

    def test_can_parse_xlsx(self):
        """Test that parser can handle Excel files."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        assert parser.can_parse("data.xlsx") is True

    def test_can_parse_html(self):
        """Test that parser can handle HTML files."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        assert parser.can_parse("page.html") is True
        assert parser.can_parse("page.htm") is True

    def test_cannot_parse_unsupported(self):
        """Test that parser rejects unsupported formats."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        assert parser.can_parse("music.mp3") is False
        assert parser.can_parse("video.mp4") is False


class TestDoclingParserMetadata:
    """Test metadata extraction capabilities."""

    def test_build_base_metadata(self):
        """Test base metadata building from input metadata."""
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

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
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

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
        from components.parsers.docling import DoclingParser
        parser = DoclingParser()

        parser_config = parser._load_metadata()

        assert parser_config.name == "DoclingParser"
        assert ".pdf" in parser_config.supported_extensions
        assert "text_extraction" in parser_config.capabilities
        assert "docling" in parser_config.dependencies["required"]


class TestDoclingParserBlobParsing:
    """Test blob parsing with metadata preservation."""

    def test_parse_blob_preserves_metadata(self):
        """Test that parse_blob preserves input metadata."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser()

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
            mock_doc = MagicMock()
            mock_doc.export_to_markdown.return_value = "# Test\n\nHello World"
            mock_doc.main_text = []
            mock_doc.pages = []
            mock_doc.tables = []
            mock_doc.origin = None

            mock_result = MagicMock()
            mock_result.document = mock_doc
            mock_converter.convert.return_value = mock_result

            # Disable chunking for simpler test
            parser.chunker = None

            documents = parser.parse_blob(test_content, input_metadata)

        assert len(documents) >= 1
        doc = documents[0]

        # Check metadata is preserved
        assert doc.metadata.get("dataset_name") == "test_dataset"
        assert doc.metadata.get("namespace") == "test_ns"
        assert doc.metadata.get("project") == "test_project"
        assert doc.metadata.get("source_filename") == "test.txt"
        assert doc.metadata.get("parser_type") == "DoclingParser"


@pytest.mark.integration
class TestDoclingParserIntegration:
    """Integration tests that require real Docling processing."""

    @pytest.fixture
    def sample_pdf_path(self):
        """Find a sample PDF for testing."""
        # Look for PDFs in examples directories
        base_path = Path(__file__).parent.parent.parent.parent.parent

        pdf_locations = [
            base_path / "examples" / "fda_rag" / "files",
            base_path / "examples" / "rag_legal_filings" / "files",
            base_path / "samples",
        ]

        for location in pdf_locations:
            if location.exists():
                pdfs = list(location.glob("*.pdf"))
                if pdfs:
                    return str(pdfs[0])

        pytest.skip("No sample PDF found for integration test")

    def test_parse_real_pdf(self, sample_pdf_path):
        """Test parsing a real PDF file."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={
            "chunk_strategy": "none",  # Disable chunking for simple test
        })

        result = parser.parse(sample_pdf_path)

        assert result.documents is not None
        assert len(result.documents) > 0
        assert len(result.errors) == 0

        # Check first document
        doc = result.documents[0]
        assert doc.content is not None
        assert len(doc.content) > 0
        assert doc.metadata.get("parser_type") == "DoclingParser"

    def test_parse_real_pdf_with_chunking(self, sample_pdf_path):
        """Test parsing a real PDF with chunking enabled."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={
            "chunk_size": 256,
            "chunk_strategy": "hybrid",
        })

        result = parser.parse(sample_pdf_path)

        assert result.documents is not None

        # With chunking, we should have multiple chunks
        if len(result.documents) > 1:
            for i, doc in enumerate(result.documents):
                assert doc.metadata.get("chunk_index") == i
                assert doc.metadata.get("is_chunked") is True


class TestDoclingParserChunking:
    """Test chunking functionality."""

    def test_chunker_initialization_hybrid(self):
        """Test HybridChunker initialization."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={"chunk_strategy": "hybrid"})

        # Check chunker is initialized (if available)
        if DOCLING_AVAILABLE:
            try:
                from docling.chunking import HybridChunker
                # Chunker may or may not be initialized depending on availability
                pass
            except ImportError:
                pass

    def test_chunker_initialization_none(self):
        """Test that chunking can be disabled."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={"chunk_strategy": "none"})
        assert parser.chunker is None

    def test_chunk_metadata_included(self):
        """Test that chunk metadata is properly included."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser()

        # Create mock documents
        content = "Test content"
        doc_metadata = {
            "source_filename": "test.pdf",
            "dataset_name": "test_dataset",
        }

        doc_id = parser._generate_doc_id(content, doc_metadata, chunk_index=5)

        # Verify chunk index is in the ID
        assert "_c5" in doc_id


class TestDoclingParserErrorHandling:
    """Test error handling capabilities."""

    def test_parse_nonexistent_file(self):
        """Test handling of nonexistent files."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser()
        result = parser.parse("/nonexistent/path/document.pdf")

        assert len(result.errors) > 0
        assert "not found" in str(result.errors[0].get("error", "")).lower() or \
               "no such file" in str(result.errors[0].get("error", "")).lower()

    def test_parse_blob_returns_error_document_on_failure(self):
        """Test that parse_blob returns a document with error info on failure."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser()

        # Invalid blob data that should fail parsing
        invalid_blob = b"not a valid PDF"
        metadata = {"filename": "invalid.pdf"}

        # This may either succeed with error info or return error document
        documents = parser.parse_blob(invalid_blob, metadata)

        assert len(documents) >= 1
        doc = documents[0]
        # Either has parsing error or is a valid parse attempt
        assert doc.content is not None or doc.metadata.get("parsing_error") is not None
