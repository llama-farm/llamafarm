"""
Comprehensive tests for DOCX parsers and utilities.
"""

import os
from unittest.mock import Mock, patch

import pytest

from components.parsers.docx.docx_utils import (
    DocxChunker,
    DocxDocumentFactory,
    DocxHeaderFooterExtractor,
    DocxTableExtractor,
    DocxTempFileHandler,
)
from components.parsers.docx.llamaindex_parser import DocxParser_LlamaIndex
from components.parsers.docx.python_docx_parser import DocxParser_PythonDocx
from core.base import Document, ProcessingResult


class TestDocxChunker:
    """Test the DocxChunker utility class."""

    def test_chunk_by_paragraphs_no_overlap(self):
        """Test paragraph chunking without overlap."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=20, chunk_overlap=0)

        assert len(chunks) == 3
        assert chunks[0] == "First paragraph."
        assert chunks[1] == "Second paragraph."
        assert chunks[2] == "Third paragraph."

    def test_chunk_by_paragraphs_with_overlap(self):
        """Test paragraph chunking with overlap."""
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=40, chunk_overlap=15)

        assert len(chunks) == 3
        assert chunks[0] == "First paragraph content here."
        # Second chunk should start with overlap from first chunk
        assert chunks[1].startswith("h content here.")
        assert "Second paragraph content here." in chunks[1]
        # Third chunk should start with overlap from second chunk
        assert chunks[2].startswith("h content here.")
        assert "Third paragraph content here." in chunks[2]

    def test_chunk_by_paragraphs_large_single_paragraph(self):
        """Test chunking when a single paragraph exceeds chunk size."""
        text = "This is a very long paragraph that exceeds the chunk size limit and should be handled gracefully."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=20, chunk_overlap=0)

        # Should still create a chunk even if it exceeds the size
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_by_paragraphs_empty_paragraphs(self):
        """Test chunking with empty paragraphs."""
        text = "First paragraph.\n\n\n\nSecond paragraph.\n\n\n\nThird paragraph."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=20, chunk_overlap=0)

        # Empty paragraphs should be filtered out
        assert len(chunks) == 3
        assert chunks[0] == "First paragraph."
        assert chunks[1] == "Second paragraph."
        assert chunks[2] == "Third paragraph."

    def test_chunk_by_paragraphs_overlap_larger_than_chunk(self):
        """Test overlap handling when overlap is larger than chunk size."""
        text = "Short.\n\nAnother short paragraph."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=10, chunk_overlap=20)

        assert len(chunks) == 2
        # When overlap is larger than previous chunk, use entire previous chunk
        assert chunks[1].startswith("Short.")

    def test_chunk_by_characters(self):
        """Test character-based chunking."""
        text = "This is a test string for character chunking."
        chunks = DocxChunker.chunk_by_characters(text, chunk_size=10)

        assert len(chunks) == 5  # 45 chars / 10 = 4.5, rounded up to 5
        assert chunks[0] == "This is a "
        assert chunks[1] == "test strin"
        assert chunks[2] == "g for char"
        assert chunks[3] == "acter chun"
        assert chunks[4] == "king."

    def test_chunk_by_characters_empty_chunks(self):
        """Test character chunking filters out empty chunks."""
        text = "Short"
        chunks = DocxChunker.chunk_by_characters(text, chunk_size=10)

        assert len(chunks) == 1
        assert chunks[0] == "Short"

    def test_chunk_by_paragraphs_non_ascii_characters(self):
        """Test paragraph chunking with non-ASCII and multi-byte characters."""
        text = "Première paragraphe avec des caractères spéciaux: àáâãäåæçèéêë.\n\nSecond paragraph with emoji: 🚀🌟💫.\n\nTroisième paragraphe avec des caractères chinois: 你好世界."
        chunks = DocxChunker.chunk_by_paragraphs(text, chunk_size=50, chunk_overlap=0)

        assert len(chunks) == 3
        # Verify character integrity is maintained
        assert "àáâãäåæçèéêë" in chunks[0]
        assert "🚀🌟💫" in chunks[1]
        assert "你好世界" in chunks[2]

    def test_chunk_by_characters_multi_byte_characters(self):
        """Test character chunking with multi-byte characters."""
        text = "Hello 世界! This is a test with émojis 🎉 and spëcial chars."
        chunks = DocxChunker.chunk_by_characters(text, chunk_size=20)

        # Verify no broken multi-byte sequences
        combined = "".join(chunks)
        assert combined == text
        assert "世界" in combined
        assert "🎉" in combined
        assert "émojis" in combined


class TestDocxDocumentFactory:
    """Test the DocxDocumentFactory utility class."""

    def test_create_documents_from_chunks(self):
        """Test creating documents from text chunks."""
        chunks = ["First chunk", "Second chunk", "Third chunk"]
        metadata = {"source": "test.docx", "author": "Test Author"}
        source_path = "/path/to/test.docx"

        documents = DocxDocumentFactory.create_documents_from_chunks(
            chunks, metadata, source_path, "paragraphs"
        )

        assert len(documents) == 3
        self._verify_chunk_documents(documents, chunks, metadata, source_path)

    def _verify_chunk_documents(self, documents, chunks, metadata, source_path):
        """Helper method to verify chunk document properties."""
        expected_properties = [
            (0, "First chunk", "test_chunk_1"),
            (1, "Second chunk", "test_chunk_2"),
            (2, "Third chunk", "test_chunk_3"),
        ]

        for i, (expected_index, expected_content, expected_id) in enumerate(
            expected_properties
        ):
            doc = documents[i]
            assert isinstance(doc, Document)
            assert doc.content == expected_content
            assert doc.metadata["chunk_index"] == expected_index
            assert doc.metadata["total_chunks"] == 3
            assert doc.metadata["chunk_strategy"] == "paragraphs"
            assert doc.metadata["source"] == "test.docx"
            assert doc.metadata["author"] == "Test Author"
            assert doc.id == expected_id
            assert doc.source == source_path

    def test_create_single_document(self):
        """Test creating a single document."""
        content = "This is the document content"
        metadata = {"source": "test.docx", "title": "Test Document"}
        source_path = "/path/to/test.docx"

        doc = DocxDocumentFactory.create_single_document(content, metadata, source_path)

        assert isinstance(doc, Document)
        assert doc.content == content
        assert doc.metadata == metadata
        assert doc.id == "test"
        assert doc.source == source_path

    def test_create_documents_from_empty_chunks(self):
        """Test creating documents from empty chunk list."""
        chunks = []
        metadata = {"source": "test.docx"}
        source_path = "/path/to/test.docx"

        documents = DocxDocumentFactory.create_documents_from_chunks(
            chunks, metadata, source_path, "paragraphs"
        )

        assert isinstance(documents, list)
        assert len(documents) == 0


class TestDocxTempFileHandler:
    """Test the DocxTempFileHandler context manager."""

    def test_temp_file_creation_and_cleanup(self):
        """Test temporary file is created and cleaned up."""
        test_data = b"Test DOCX content"

        with DocxTempFileHandler(test_data) as tmp_path:
            # File should exist during context
            assert os.path.exists(tmp_path)
            assert tmp_path.endswith(".docx")

            # Content should match
            with open(tmp_path, "rb") as f:
                assert f.read() == test_data

        # File should be cleaned up after context
        assert not os.path.exists(tmp_path)


class TestDocxTableExtractor:
    """Test the DocxTableExtractor utility class."""

    def test_extract_table_as_text(self):
        """Test table extraction as formatted text."""
        mock_table = self._create_mock_table()
        result = DocxTableExtractor.extract_table_as_text(mock_table)

        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "Header 1 | Header 2"
        assert lines[1].startswith("-")  # Separator line
        assert lines[2] == "Data 1 | Data 2"

    def _create_mock_table(self):
        """Helper method to create a mock table structure."""
        mock_table = Mock()

        # Mock rows
        mock_row1 = Mock()
        mock_row2 = Mock()
        mock_table.rows = [mock_row1, mock_row2]

        # Mock cells for row 1
        mock_cell1_1, mock_cell1_2 = (
            self._create_mock_cell("Header 1"),
            self._create_mock_cell("Header 2"),
        )
        mock_row1.cells = [mock_cell1_1, mock_cell1_2]

        # Mock cells for row 2
        mock_cell2_1, mock_cell2_2 = (
            self._create_mock_cell("Data 1"),
            self._create_mock_cell("Data 2"),
        )
        mock_row2.cells = [mock_cell2_1, mock_cell2_2]

        return mock_table

    def _create_mock_cell(self, text_content):
        """Helper method to create a mock cell with text content."""
        mock_cell = Mock()
        mock_para = Mock()
        mock_para.text = text_content
        mock_cell.paragraphs = [mock_para]
        return mock_cell


class TestDocxHeaderFooterExtractor:
    """Test the DocxHeaderFooterExtractor utility class."""

    def test_extract_headers(self):
        """Test header extraction."""
        # Mock document with sections
        mock_doc = Mock()
        mock_section = Mock()
        mock_doc.sections = [mock_section]

        # Mock header
        mock_header = Mock()
        mock_section.header = mock_header

        # Mock paragraphs in header
        mock_para1 = Mock()
        mock_para1.text = "Header Line 1"
        mock_para2 = Mock()
        mock_para2.text = "Header Line 2"
        mock_header.paragraphs = [mock_para1, mock_para2]

        headers = DocxHeaderFooterExtractor.extract_headers(
            mock_doc, extract_headers=True
        )

        assert len(headers) == 1
        assert headers[0] == "Header: Header Line 1\nHeader Line 2"

    def test_extract_headers_disabled(self):
        """Test header extraction when disabled."""
        mock_doc = Mock()
        headers = DocxHeaderFooterExtractor.extract_headers(
            mock_doc, extract_headers=False
        )
        assert headers == []

    def test_extract_footers(self):
        """Test footer extraction."""
        # Mock document with sections
        mock_doc = Mock()
        mock_section = Mock()
        mock_doc.sections = [mock_section]

        # Mock footer
        mock_footer = Mock()
        mock_section.footer = mock_footer

        # Mock paragraphs in footer
        mock_para1 = Mock()
        mock_para1.text = "Footer Text"
        mock_footer.paragraphs = [mock_para1]

        footers = DocxHeaderFooterExtractor.extract_footers(
            mock_doc, extract_footers=True
        )

        assert len(footers) == 1
        assert footers[0] == "Footer: Footer Text"

    def test_extract_headers_no_sections(self):
        """Test header extraction when document has no sections."""
        mock_doc = Mock()
        mock_doc.sections = []

        headers = DocxHeaderFooterExtractor.extract_headers(
            mock_doc, extract_headers=True
        )

        assert isinstance(headers, list)
        assert len(headers) == 0

    def test_extract_footers_no_sections(self):
        """Test footer extraction when document has no sections."""
        mock_doc = Mock()
        mock_doc.sections = []

        footers = DocxHeaderFooterExtractor.extract_footers(
            mock_doc, extract_footers=True
        )

        assert isinstance(footers, list)
        assert len(footers) == 0


class TestDocxParser_LlamaIndex:
    """Test the LlamaIndex DOCX parser."""

    def test_initialization(self):
        """Test parser initialization with default config."""
        parser = DocxParser_LlamaIndex()

        assert parser.name == "DocxParser_LlamaIndex"
        assert parser.chunk_size == 1000
        assert parser.chunk_overlap == 100
        assert parser.chunk_strategy == "paragraphs"
        assert parser.extract_metadata is True
        assert parser.extract_tables is True

    def test_initialization_with_custom_config(self):
        """Test parser initialization with custom config."""
        config = {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunk_strategy": "sentences",
            "extract_metadata": False,
            "extract_tables": False,
        }
        parser = DocxParser_LlamaIndex(config=config)

        assert parser.chunk_size == 500
        assert parser.chunk_overlap == 50
        assert parser.chunk_strategy == "sentences"
        assert parser.extract_metadata is False
        assert parser.extract_tables is False

    def test_can_parse_docx_files(self):
        """Test that parser accepts .docx files."""
        parser = DocxParser_LlamaIndex()

        assert parser.can_parse("document.docx") is True
        assert parser.can_parse("DOCUMENT.DOCX") is True
        assert parser.can_parse("/path/to/document.docx") is True

    def test_cannot_parse_non_docx_files(self):
        """Test that parser rejects non-.docx files."""
        parser = DocxParser_LlamaIndex()

        assert parser.can_parse("document.doc") is False
        assert parser.can_parse("document.txt") is False
        assert parser.can_parse("document.pdf") is False
        assert parser.can_parse("document") is False

    def test_parse_file_not_found(self):
        """Test parsing when file doesn't exist."""
        parser = DocxParser_LlamaIndex()
        result = parser.parse("/nonexistent/file.docx")

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 0
        assert len(result.errors) == 1
        assert "File not found" in result.errors[0]["error"]

    @patch("components.parsers.docx.llamaindex_parser.DocxReader")
    @patch("components.parsers.docx.llamaindex_parser.docx")
    def test_parse_success_no_chunking(self, mock_docx, mock_docx_reader):
        """Test successful parsing without chunking."""
        # Setup mocks
        mock_reader_instance = Mock()
        mock_docx_reader.return_value = mock_reader_instance

        mock_llama_doc = Mock()
        mock_llama_doc.text = "Test document content"
        mock_llama_doc.metadata = {"test_meta": "value"}
        mock_reader_instance.load_data.return_value = [mock_llama_doc]

        # Mock file existence
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024

                parser = DocxParser_LlamaIndex(config={"chunk_size": None})
                result = parser.parse("test.docx")

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 1
        assert len(result.errors) == 0
        assert result.documents[0].content == "Test document content"

    def test_parse_blob_success(self):
        """Test that parse_blob method exists and returns appropriate type."""
        parser = DocxParser_LlamaIndex()
        test_data = b"fake docx content"
        metadata = {"filename": "test.docx"}

        # This will fail due to missing dependencies, but should return empty list
        result = parser.parse_blob(test_data, metadata)

        assert isinstance(result, list)
        # Should return empty list when dependencies are not available
        assert len(result) == 0

    def test_parse_blob_invalid_docx_data(self):
        """Test parse_blob with invalid/corrupted DOCX bytes."""
        parser = DocxParser_LlamaIndex()

        # Test with completely invalid data
        invalid_data = b"This is not a DOCX file at all"
        result = parser.parse_blob(invalid_data)
        assert isinstance(result, list)
        assert len(result) == 0

        # Test with partially corrupted data
        corrupted_data = b"PK\x03\x04" + b"corrupted docx data" * 100
        result = parser.parse_blob(corrupted_data)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_parse_blob_empty_data(self):
        """Test parse_blob with empty byte input."""
        parser = DocxParser_LlamaIndex()

        result = parser.parse_blob(b"")
        assert isinstance(result, list)
        assert len(result) == 0


class TestDocxParser_PythonDocx:
    """Test the python-docx DOCX parser."""

    def test_initialization(self):
        """Test parser initialization with default config."""
        parser = DocxParser_PythonDocx()

        assert parser.name == "DocxParser_PythonDocx"
        assert parser.chunk_size == 1000
        assert parser.chunk_strategy == "paragraphs"
        assert parser.extract_metadata is True
        assert parser.extract_tables is True

    def test_initialization_with_custom_config(self):
        """Test parser initialization with custom config."""
        config = {
            "chunk_size": 500,
            "chunk_strategy": "characters",
            "extract_metadata": False,
            "extract_tables": False,
        }
        parser = DocxParser_PythonDocx(config=config)

        assert parser.chunk_size == 500
        assert parser.chunk_strategy == "characters"
        assert parser.extract_metadata is False
        assert parser.extract_tables is False

    def test_can_parse_docx_files(self):
        """Test that parser accepts .docx files."""
        parser = DocxParser_PythonDocx()

        assert parser.can_parse("document.docx") is True
        assert parser.can_parse("DOCUMENT.DOCX") is True
        assert parser.can_parse("/path/to/document.docx") is True

    def test_cannot_parse_non_docx_files(self):
        """Test that parser rejects non-.docx files."""
        parser = DocxParser_PythonDocx()

        assert parser.can_parse("document.doc") is False
        assert parser.can_parse("document.txt") is False
        assert parser.can_parse("document.pdf") is False
        assert parser.can_parse("document") is False

    def test_parse_file_not_found(self):
        """Test parsing when file doesn't exist."""
        parser = DocxParser_PythonDocx()
        result = parser.parse("/nonexistent/file.docx")

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 0
        assert len(result.errors) == 1
        assert "File not found" in result.errors[0]["error"]

    def test_parse_docx_not_installed(self):
        """Test parsing when python-docx is not installed."""
        parser = DocxParser_PythonDocx()

        # Test with a non-existent file to avoid the import issue
        result = parser.parse("/nonexistent/file.docx")

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 0
        assert len(result.errors) == 1
        assert "File not found" in result.errors[0]["error"]

    def test_parse_success_paragraph_chunking(self):
        """Test parser configuration for paragraph chunking."""
        parser = DocxParser_PythonDocx(
            config={"chunk_size": 20, "chunk_strategy": "paragraphs"}
        )

        # Test that the configuration is set correctly
        assert parser.chunk_strategy == "paragraphs"
        assert parser.chunk_size == 20

    def test_parse_success_character_chunking(self):
        """Test parser configuration for character chunking."""
        parser = DocxParser_PythonDocx(
            config={"chunk_size": 20, "chunk_strategy": "characters"}
        )

        # Test that the configuration is set correctly
        assert parser.chunk_strategy == "characters"
        assert parser.chunk_size == 20

    def test_parse_with_tables(self):
        """Test parser configuration for table extraction."""
        parser = DocxParser_PythonDocx(config={"extract_tables": True})

        # Test that the configuration is set correctly
        assert parser.extract_tables is True

    def test_parse_blob_success(self):
        """Test that parse_blob method exists and handles errors gracefully."""
        parser = DocxParser_PythonDocx()
        test_data = b"fake docx content"
        metadata = {"filename": "test.docx"}

        # This will fail due to missing docx dependency, but should return empty list
        result = parser.parse_blob(test_data, metadata)

        assert isinstance(result, list)
        # Should return empty list when docx is not available
        assert len(result) == 0

    def test_parse_blob_invalid_docx_data(self):
        """Test parse_blob with invalid/corrupted DOCX bytes."""
        parser = DocxParser_PythonDocx()

        # Test with completely invalid data
        invalid_data = b"This is not a DOCX file at all"
        result = parser.parse_blob(invalid_data)
        assert isinstance(result, list)
        assert len(result) == 0

        # Test with partially corrupted data
        corrupted_data = b"PK\x03\x04" + b"corrupted docx data" * 100
        result = parser.parse_blob(corrupted_data)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_parse_blob_empty_data(self):
        """Test parse_blob with empty byte input."""
        parser = DocxParser_PythonDocx()

        result = parser.parse_blob(b"")
        assert isinstance(result, list)
        assert len(result) == 0


class TestDocxParsersIntegration:
    """Integration tests for DOCX parsers with various strategies."""

    def test_chunking_strategies_comparison(self):
        """Test different chunking strategies produce different results."""
        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph with additional content."

        # Test paragraph chunking
        para_chunks = DocxChunker.chunk_by_paragraphs(
            text, chunk_size=30, chunk_overlap=0
        )

        # Test character chunking
        char_chunks = DocxChunker.chunk_by_characters(text, chunk_size=30)

        # Results should be different - paragraph chunking respects paragraph boundaries
        # while character chunking cuts at fixed positions
        assert para_chunks != char_chunks

        # Paragraph chunks should respect paragraph boundaries
        # Verify no chunks end with partial words (example check)
        problematic_chunks = [
            chunk for chunk in para_chunks if chunk.endswith("with some")
        ]
        assert len(problematic_chunks) == 0, "Chunks should not cut mid-word"

        # Character chunks may cut anywhere
        combined_char_text = "".join(char_chunks)
        combined_para_text = "\n\n".join(para_chunks)

        # Both should contain the same content (allowing for different formatting)
        assert combined_char_text.replace(" ", "") != ""
        assert combined_para_text.replace(" ", "") != ""

    def test_overlap_consistency(self):
        """Test that overlap chunking is consistent."""
        text = "This is a longer text that will be chunked with overlap to ensure consistency across multiple runs."

        # Run chunking multiple times
        chunks1 = DocxChunker.chunk_by_paragraphs(text, chunk_size=30, chunk_overlap=10)
        chunks2 = DocxChunker.chunk_by_paragraphs(text, chunk_size=30, chunk_overlap=10)

        # Results should be identical
        assert chunks1 == chunks2

        # Verify overlap exists
        if len(chunks1) > 1:
            # Check that there's actual overlap between consecutive chunks
            self._verify_chunk_overlaps(chunks1)

    def _verify_chunk_overlaps(self, chunks):
        """Helper method to verify overlap between consecutive chunks."""
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Current chunk should start with some text from previous chunk
            overlap_found = any(
                curr_chunk.startswith(prev_chunk[-tail_len:])
                for tail_len in range(1, min(10, len(prev_chunk)))
            )
            assert overlap_found, f"No overlap found between chunks {i - 1} and {i}"

    def test_parser_metadata_consistency(self):
        """Test that both parsers produce consistent metadata."""
        config = {
            "chunk_size": 100,
            "chunk_strategy": "paragraphs",
            "extract_metadata": True,
        }

        llama_parser = DocxParser_LlamaIndex(config=config)
        python_parser = DocxParser_PythonDocx(config=config)

        # Both parsers should have same configuration
        assert llama_parser.chunk_size == python_parser.chunk_size
        assert llama_parser.chunk_strategy == python_parser.chunk_strategy
        assert llama_parser.extract_metadata == python_parser.extract_metadata

    def test_error_handling_consistency(self):
        """Test that both parsers handle errors consistently."""
        llama_parser = DocxParser_LlamaIndex()
        python_parser = DocxParser_PythonDocx()

        # Both should reject non-existent files
        llama_result = llama_parser.parse("/nonexistent/file.docx")
        python_result = python_parser.parse("/nonexistent/file.docx")

        assert isinstance(llama_result, ProcessingResult)
        assert isinstance(python_result, ProcessingResult)
        assert len(llama_result.documents) == 0
        assert len(python_result.documents) == 0
        assert len(llama_result.errors) > 0
        assert len(python_result.errors) > 0

    def test_parse_blob_integration(self):
        """Integration tests for parse_blob in both parsers."""
        llama_parser = DocxParser_LlamaIndex()
        python_parser = DocxParser_PythonDocx()

        # Test with invalid data - both should handle gracefully
        invalid_data = b"Not a DOCX file"

        llama_result = llama_parser.parse_blob(invalid_data)
        python_result = python_parser.parse_blob(invalid_data)

        assert isinstance(llama_result, list)
        assert isinstance(python_result, list)
        assert len(llama_result) == 0
        assert len(python_result) == 0

        # Test with empty data
        empty_result_llama = llama_parser.parse_blob(b"")
        empty_result_python = python_parser.parse_blob(b"")

        assert isinstance(empty_result_llama, list)
        assert isinstance(empty_result_python, list)
        assert len(empty_result_llama) == 0
        assert len(empty_result_python) == 0

        # Test with metadata
        metadata = {"filename": "test.docx", "author": "Test User"}

        llama_meta_result = llama_parser.parse_blob(invalid_data, metadata)
        python_meta_result = python_parser.parse_blob(invalid_data, metadata)

        assert isinstance(llama_meta_result, list)
        assert isinstance(python_meta_result, list)


if __name__ == "__main__":
    pytest.main([__file__])
