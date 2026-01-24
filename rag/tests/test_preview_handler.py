"""Unit tests for PreviewHandler - TDD Red Phase.

All tests written FIRST and will fail until implementation is complete.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.base import Document


class TestPreviewHandler:
    """Unit tests for PreviewHandler."""

    def test_preview_handler_can_be_instantiated(self):
        """PreviewHandler can be instantiated with a blob_processor."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        handler = PreviewHandler(blob_processor=mock_blob_processor)
        assert handler is not None
        assert handler.blob_processor is mock_blob_processor

    def test_generate_preview_uses_same_blob_processor(self):
        """Preview must use exact same process_blob() as ingestion."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Test chunk 1", metadata={"chunk_index": 0}),
            Document(content="Test chunk 2", metadata={"chunk_index": 1}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        file_data = b"Test chunk 1Test chunk 2"
        metadata = {"filename": "test.txt"}

        result = handler.generate_preview(file_data, metadata)

        # Verify blob_processor.process_blob was called with same args
        mock_blob_processor.process_blob.assert_called_once_with(file_data, metadata)
        assert result is not None

    def test_generate_preview_returns_preview_result(self):
        """generate_preview returns a PreviewResult object."""
        from core.base import PreviewResult
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Hello world.", metadata={"parser": "TextParser"}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Hello world.",
            {"filename": "test.txt"}
        )

        assert isinstance(result, PreviewResult)
        assert result.original_text is not None
        assert result.chunks is not None
        assert result.total_chunks >= 0

    def test_compute_positions_sequential_chunks(self):
        """Position computation for non-overlapping sequential chunks."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        handler = PreviewHandler(blob_processor=mock_blob_processor)

        full_text = "Hello world. This is a test."
        documents = [
            Document(content="Hello world.", metadata={}),
            Document(content="This is a test.", metadata={}),
        ]

        positions = handler._compute_positions(full_text, documents)

        assert len(positions) == 2
        # First chunk: "Hello world." starts at 0, ends at 12
        assert positions[0].start_position == 0
        assert positions[0].end_position == 12
        assert positions[0].content == "Hello world."
        # Second chunk: "This is a test." starts at 13, ends at 28
        # (there's a space at position 12)
        assert positions[1].start_position == 13
        assert positions[1].end_position == 28
        assert positions[1].content == "This is a test."

    def test_compute_positions_with_overlap(self):
        """Position computation accounts for chunk overlap."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        handler = PreviewHandler(blob_processor=mock_blob_processor)

        full_text = "ABCDEFGHIJ"
        # chunk_size=5, overlap=2 → chunks: "ABCDE", "DEFGH", "GHIJ"
        documents = [
            Document(content="ABCDE", metadata={"chunk_overlap": 2}),
            Document(content="DEFGH", metadata={"chunk_overlap": 2}),
            Document(content="GHIJ", metadata={"chunk_overlap": 2}),
        ]

        positions = handler._compute_positions(full_text, documents)

        assert len(positions) == 3
        # First chunk: "ABCDE" at positions 0-5
        assert positions[0].start_position == 0
        assert positions[0].end_position == 5
        # Second chunk: "DEFGH" at positions 3-8 (overlaps with first)
        assert positions[1].start_position == 3
        assert positions[1].end_position == 8
        # Third chunk: "GHIJ" at positions 6-10 (overlaps with second)
        assert positions[2].start_position == 6
        assert positions[2].end_position == 10

    def test_compute_positions_handles_missing_chunk(self):
        """Gracefully handle chunk not found in full text."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        handler = PreviewHandler(blob_processor=mock_blob_processor)

        full_text = "Hello world"
        # Chunk that doesn't exist in full text (edge case: modified by extractor)
        documents = [
            Document(content="Hello world", metadata={}),
            Document(content="XYZ not found", metadata={}),
        ]

        positions = handler._compute_positions(full_text, documents)

        assert len(positions) == 2
        # First chunk found
        assert positions[0].start_position == 0
        assert positions[0].end_position == 11
        # Second chunk not found - should have position -1
        assert positions[1].start_position == -1
        assert positions[1].end_position == -1

    def test_preview_with_chunk_size_override(self):
        """Override chunk_size affects resulting chunks."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Small", metadata={}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Test content",
            {"filename": "test.txt"},
            chunk_size_override=100,
        )

        # Verify the metadata was modified with the override
        call_args = mock_blob_processor.process_blob.call_args
        metadata_passed = call_args[0][1]
        assert metadata_passed.get("chunk_size") == 100

    def test_preview_with_chunk_overlap_override(self):
        """Override chunk_overlap affects resulting chunks."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Test", metadata={}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Test content",
            {"filename": "test.txt"},
            chunk_overlap_override=50,
        )

        call_args = mock_blob_processor.process_blob.call_args
        metadata_passed = call_args[0][1]
        assert metadata_passed.get("chunk_overlap") == 50

    def test_preview_with_chunk_strategy_override(self):
        """Override chunk_strategy affects chunking method."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Test", metadata={}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Test content",
            {"filename": "test.txt"},
            chunk_strategy_override="sentences",
        )

        call_args = mock_blob_processor.process_blob.call_args
        metadata_passed = call_args[0][1]
        assert metadata_passed.get("chunk_strategy") == "sentences"

    def test_preview_result_statistics(self):
        """PreviewResult contains correct statistics."""
        from core.base import PreviewResult
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="A" * 100, metadata={}),
            Document(content="B" * 150, metadata={}),
            Document(content="C" * 200, metadata={}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"A" * 100 + b"B" * 150 + b"C" * 200,
            {"filename": "test.txt"},
        )

        assert result.total_chunks == 3
        # Average chunk size: (100 + 150 + 200) / 3 = 150
        assert result.avg_chunk_size == 150.0

    def test_preview_result_contains_file_info(self):
        """PreviewResult contains file metadata."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Test", metadata={"parser": "TextParser"}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Test content here",
            {"filename": "test.txt", "content_type": "text/plain"},
        )

        assert result.file_info["filename"] == "test.txt"
        assert result.file_info["size"] == len(b"Test content here")

    def test_preview_result_contains_parser_info(self):
        """PreviewResult contains parser information from documents."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Test", metadata={"parser": "PDFParser_LlamaIndex"}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Test content",
            {"filename": "test.pdf"},
        )

        assert result.parser_used == "PDFParser_LlamaIndex"

    def test_preview_chunks_contain_word_and_char_counts(self):
        """Preview chunks include word count and character count."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Hello world test", metadata={}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        result = handler.generate_preview(
            b"Hello world test",
            {"filename": "test.txt"},
        )

        assert len(result.chunks) == 1
        chunk = result.chunks[0]
        assert chunk.char_count == 16  # "Hello world test" = 16 chars
        assert chunk.word_count == 3   # 3 words

    def test_get_full_text_extracts_text_from_bytes(self):
        """_get_full_text extracts readable text from file bytes."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        handler = PreviewHandler(blob_processor=mock_blob_processor)

        # Simple text file
        file_data = b"This is the full text content."
        metadata = {"filename": "test.txt"}
        documents = []  # Not needed for text files

        full_text = handler._get_full_text(file_data, metadata, documents)

        assert full_text == "This is the full text content."

    def test_preview_with_empty_file_raises_error(self):
        """Preview raises appropriate error for empty file."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = []  # No documents

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        with pytest.raises(ValueError, match="No chunks"):
            handler.generate_preview(b"", {"filename": "empty.txt"})

    def test_preview_handles_binary_file(self):
        """Preview handles binary files like PDF by using parser output."""
        from core.preview_handler import PreviewHandler

        mock_blob_processor = Mock()
        mock_blob_processor.process_blob.return_value = [
            Document(content="Extracted PDF text", metadata={"parser": "PDFParser"}),
        ]

        handler = PreviewHandler(blob_processor=mock_blob_processor)

        # Binary PDF-like data
        file_data = b"%PDF-1.4 binary content"
        metadata = {"filename": "test.pdf", "content_type": "application/pdf"}

        result = handler.generate_preview(file_data, metadata)

        # For binary files, original_text should come from parsed content
        assert "Extracted PDF text" in result.original_text
        assert result.chunks[0].content == "Extracted PDF text"


class TestPreviewChunk:
    """Tests for PreviewChunk dataclass."""

    def test_preview_chunk_creation(self):
        """PreviewChunk can be created with required fields."""
        from core.base import PreviewChunk

        chunk = PreviewChunk(
            chunk_index=0,
            content="Test content",
            start_position=0,
            end_position=12,
        )

        assert chunk.chunk_index == 0
        assert chunk.content == "Test content"
        assert chunk.start_position == 0
        assert chunk.end_position == 12
        assert chunk.metadata == {}  # default

    def test_preview_chunk_with_metadata(self):
        """PreviewChunk stores metadata."""
        from core.base import PreviewChunk

        chunk = PreviewChunk(
            chunk_index=1,
            content="Content",
            start_position=10,
            end_position=17,
            metadata={"parser": "TextParser", "custom_field": "value"},
        )

        assert chunk.metadata["parser"] == "TextParser"
        assert chunk.metadata["custom_field"] == "value"

    def test_preview_chunk_char_count(self):
        """PreviewChunk calculates character count."""
        from core.base import PreviewChunk

        chunk = PreviewChunk(
            chunk_index=0,
            content="Hello World!",  # 12 characters
            start_position=0,
            end_position=12,
        )

        assert chunk.char_count == 12

    def test_preview_chunk_word_count(self):
        """PreviewChunk calculates word count."""
        from core.base import PreviewChunk

        chunk = PreviewChunk(
            chunk_index=0,
            content="Hello world this is a test",  # 6 words
            start_position=0,
            end_position=26,
        )

        assert chunk.word_count == 6


class TestPreviewResult:
    """Tests for PreviewResult dataclass."""

    def test_preview_result_creation(self):
        """PreviewResult can be created with required fields."""
        from core.base import PreviewChunk, PreviewResult

        chunks = [
            PreviewChunk(
                chunk_index=0,
                content="Test",
                start_position=0,
                end_position=4,
            )
        ]

        result = PreviewResult(
            original_text="Test content",
            chunks=chunks,
            file_info={"filename": "test.txt", "size": 12},
            parser_used="TextParser",
            chunk_strategy="characters",
            chunk_size=500,
            chunk_overlap=50,
            total_chunks=1,
        )

        assert result.original_text == "Test content"
        assert len(result.chunks) == 1
        assert result.total_chunks == 1
        assert result.parser_used == "TextParser"

    def test_preview_result_defaults(self):
        """PreviewResult has correct defaults."""
        from core.base import PreviewResult

        result = PreviewResult(
            original_text="Test",
            chunks=[],
            file_info={},
            parser_used="Unknown",
            chunk_strategy="characters",
            chunk_size=500,
            chunk_overlap=50,
            total_chunks=0,
        )

        assert result.warnings == []
        assert result.avg_chunk_size == 0.0
        assert result.total_size_with_overlaps == 0

    def test_preview_result_to_dict(self):
        """PreviewResult can be serialized to dictionary."""
        from core.base import PreviewChunk, PreviewResult

        chunks = [
            PreviewChunk(
                chunk_index=0,
                content="Test",
                start_position=0,
                end_position=4,
            )
        ]

        result = PreviewResult(
            original_text="Test content",
            chunks=chunks,
            file_info={"filename": "test.txt"},
            parser_used="TextParser",
            chunk_strategy="characters",
            chunk_size=500,
            chunk_overlap=50,
            total_chunks=1,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["original_text"] == "Test content"
        assert data["total_chunks"] == 1
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["content"] == "Test"

    def test_preview_result_avg_chunk_size_calculation(self):
        """PreviewResult correctly calculates average chunk size."""
        from core.base import PreviewChunk, PreviewResult

        chunks = [
            PreviewChunk(chunk_index=0, content="A" * 100, start_position=0, end_position=100),
            PreviewChunk(chunk_index=1, content="B" * 200, start_position=100, end_position=300),
            PreviewChunk(chunk_index=2, content="C" * 300, start_position=300, end_position=600),
        ]

        result = PreviewResult(
            original_text="...",
            chunks=chunks,
            file_info={},
            parser_used="Test",
            chunk_strategy="characters",
            chunk_size=500,
            chunk_overlap=0,
            total_chunks=3,
        )

        # Average: (100 + 200 + 300) / 3 = 200
        assert result.avg_chunk_size == 200.0

    def test_preview_result_total_size_with_overlaps(self):
        """PreviewResult correctly calculates total size including overlaps."""
        from core.base import PreviewChunk, PreviewResult

        # Three chunks of 100 chars each
        chunks = [
            PreviewChunk(chunk_index=0, content="A" * 100, start_position=0, end_position=100),
            PreviewChunk(chunk_index=1, content="B" * 100, start_position=80, end_position=180),
            PreviewChunk(chunk_index=2, content="C" * 100, start_position=160, end_position=260),
        ]

        result = PreviewResult(
            original_text="...",
            chunks=chunks,
            file_info={},
            parser_used="Test",
            chunk_strategy="characters",
            chunk_size=100,
            chunk_overlap=20,
            total_chunks=3,
        )

        # Total size with overlaps: 100 + 100 + 100 = 300
        assert result.total_size_with_overlaps == 300
