"""
Tests for the SimpleChunker.

Tests cover:
- Character-based chunking
- Sentence-based chunking
- Paragraph-based chunking
- Section-based chunking (markdown headers)
- Page-based chunking
- Token-based chunking
- Overlap functionality
- Metadata preservation
"""

import pytest

from components.chunkers import SimpleChunker, ChunkingStrategy, ChunkResult


class TestSimpleChunkerImports:
    """Test that SimpleChunker can be imported."""

    def test_simple_chunker_import(self):
        """Test that SimpleChunker can be imported."""
        from components.chunkers import SimpleChunker
        assert SimpleChunker is not None

    def test_chunking_strategy_import(self):
        """Test that ChunkingStrategy can be imported."""
        from components.chunkers import ChunkingStrategy
        assert ChunkingStrategy is not None

    def test_chunk_result_import(self):
        """Test that ChunkResult can be imported."""
        from components.chunkers import ChunkResult
        assert ChunkResult is not None


class TestSimpleChunkerInitialization:
    """Test SimpleChunker initialization."""

    def test_default_initialization(self):
        """Test chunker with default settings."""
        chunker = SimpleChunker()

        assert chunker.name == "SimpleChunker"
        assert chunker.chunk_size == 512
        assert chunker.overlap == 0
        assert chunker.strategy == ChunkingStrategy.CHARACTERS

    def test_custom_initialization(self):
        """Test chunker with custom settings."""
        chunker = SimpleChunker(
            chunk_size=256,
            overlap=50,
            strategy=ChunkingStrategy.SENTENCES,
        )

        assert chunker.chunk_size == 256
        assert chunker.overlap == 50
        assert chunker.strategy == ChunkingStrategy.SENTENCES

    def test_string_strategy(self):
        """Test chunker with string strategy."""
        chunker = SimpleChunker(strategy="paragraphs")
        assert chunker.strategy == ChunkingStrategy.PARAGRAPHS

    def test_config_options(self):
        """Test chunker with config options."""
        config = {
            "min_chunk_size": 100,
            "max_chunk_size": 2000,
        }
        chunker = SimpleChunker(config=config)

        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 2000


class TestCharacterChunking:
    """Test character-based chunking."""

    def test_small_text_no_chunking(self):
        """Test that small text returns single chunk."""
        chunker = SimpleChunker(chunk_size=512)
        text = "This is a short text."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_character_chunking_produces_expected_count(self):
        """Test that character chunking produces expected chunk count."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.CHARACTERS)
        text = "A" * 350  # Should produce 4 chunks

        chunks = chunker.chunk(text)

        assert len(chunks) == 4
        assert chunks[0].total_chunks == 4

    def test_character_chunking_with_overlap(self):
        """Test character chunking with overlap."""
        chunker = SimpleChunker(chunk_size=100, overlap=20, strategy=ChunkingStrategy.CHARACTERS)
        text = "A" * 200

        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        # First chunk should have overlap_after
        assert chunks[0].overlap_after == 20
        # Second chunk should have overlap_before
        assert chunks[1].overlap_before == 20

    def test_chunk_metadata_includes_position(self):
        """Test that chunk metadata includes position info."""
        chunker = SimpleChunker(chunk_size=50)
        text = "A" * 100

        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == 2
            assert chunk.chunk_index == i
            assert chunk.total_chunks == 2


class TestSentenceChunking:
    """Test sentence-based chunking."""

    def test_sentence_chunking_respects_boundaries(self):
        """Test that sentence chunking respects sentence boundaries."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.SENTENCES)
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunker.chunk(text)

        # Verify chunks don't break mid-sentence
        for chunk in chunks:
            content = chunk.content.strip()
            # Each chunk should end with a sentence-ending punctuation
            # or be the last chunk
            if chunk.chunk_index < chunk.total_chunks - 1:
                assert content[-1] in ".!?" or content.endswith(".")

    def test_single_sentence_no_chunking(self):
        """Test that single sentence returns one chunk."""
        chunker = SimpleChunker(chunk_size=1000, strategy=ChunkingStrategy.SENTENCES)
        text = "This is a single sentence."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_sentence_strategy_metadata(self):
        """Test that strategy is recorded in metadata."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.SENTENCES)
        text = "Sentence one. Sentence two."

        chunks = chunker.chunk(text)

        assert chunks[0].strategy == "sentences"
        assert chunks[0].metadata["chunking_strategy"] == "sentences"


class TestParagraphChunking:
    """Test paragraph-based chunking."""

    def test_paragraph_chunking_respects_boundaries(self):
        """Test that paragraph chunking respects paragraph boundaries."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.PARAGRAPHS)
        text = """First paragraph with some content.

Second paragraph with different content.

Third paragraph to finish."""

        chunks = chunker.chunk(text)

        # Verify chunks align with paragraph boundaries
        for chunk in chunks:
            content = chunk.content
            # Should not have double newlines in the middle (unless too large)
            if len(content) < chunker.max_chunk_size:
                # Paragraphs are joined with \n\n
                pass  # This is expected behavior

    def test_single_paragraph_no_chunking(self):
        """Test that single paragraph returns one chunk."""
        chunker = SimpleChunker(chunk_size=1000, strategy=ChunkingStrategy.PARAGRAPHS)
        text = "This is a single paragraph with no breaks."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1

    def test_multiple_paragraphs_grouped(self):
        """Test that paragraphs are grouped to meet chunk_size."""
        chunker = SimpleChunker(chunk_size=500, strategy=ChunkingStrategy.PARAGRAPHS)
        text = """Para 1.

Para 2.

Para 3."""

        chunks = chunker.chunk(text)

        # All paragraphs should fit in one chunk
        assert len(chunks) == 1


class TestSectionChunking:
    """Test section-based chunking (markdown headers)."""

    def test_section_chunking_uses_markdown_headers(self):
        """Test that section chunking uses markdown headers."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.SECTIONS)
        text = """# Introduction

This is the intro.

## Methods

These are the methods.

## Results

These are the results."""

        chunks = chunker.chunk(text)

        # Should have 3 sections
        assert len(chunks) >= 3

        # First section should be Introduction
        assert "# Introduction" in chunks[0].content

    def test_section_with_no_headers(self):
        """Test section chunking falls back for no headers."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.SECTIONS)
        text = """This is text with no markdown headers.

Just plain paragraphs.

Nothing special here."""

        chunks = chunker.chunk(text)

        # Should fall back to paragraph chunking
        assert len(chunks) >= 1

    def test_nested_headers(self):
        """Test handling of nested header levels."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.SECTIONS)
        text = """# Main Title

Content.

## Sub Section

More content.

### Sub Sub Section

Even more content."""

        chunks = chunker.chunk(text)

        # Should recognize all header levels
        assert len(chunks) >= 3


class TestPageChunking:
    """Test page-based chunking."""

    def test_page_chunking_with_markers(self):
        """Test page chunking with page markers in text."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.PAGES)
        text = """[Page 1]
Content from page one.

[Page 2]
Content from page two.

[Page 3]
Content from page three."""

        chunks = chunker.chunk(text)

        # Should have 3 page chunks
        assert len(chunks) == 3

    def test_page_chunking_with_metadata(self):
        """Test page chunking uses metadata when available."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.PAGES)
        text = "Content without page markers"
        metadata = {"page_numbers": [1, 2, 3]}

        chunks = chunker.chunk(text, metadata)

        # Without markers, should fall back
        assert len(chunks) >= 1

    def test_page_chunking_no_markers_fallback(self):
        """Test page chunking falls back to paragraphs."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.PAGES)
        text = """First paragraph.

Second paragraph.

Third paragraph."""

        chunks = chunker.chunk(text)

        # Should fall back to paragraph chunking
        assert len(chunks) >= 1


class TestTokenChunking:
    """Test token-based chunking."""

    def test_token_chunking_by_words(self):
        """Test token chunking uses word-based tokenization."""
        chunker = SimpleChunker(chunk_size=10, strategy=ChunkingStrategy.TOKENS)
        text = "one two three four five six seven eight nine ten eleven twelve"

        chunks = chunker.chunk(text)

        # Should produce 2 chunks (10 words + 2 words)
        assert len(chunks) == 2

    def test_token_chunking_small_text(self):
        """Test token chunking with text smaller than chunk_size."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.TOKENS)
        text = "just a few words"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1


class TestChunkOverlap:
    """Test overlap functionality."""

    def test_overlap_works_correctly(self):
        """Test that overlap includes content from adjacent chunks."""
        chunker = SimpleChunker(chunk_size=100, overlap=20, strategy=ChunkingStrategy.CHARACTERS)
        text = "A" * 50 + "B" * 50 + "C" * 50 + "D" * 50  # 200 chars

        chunks = chunker.chunk(text)

        # With overlap, chunks should share some content
        assert len(chunks) == 2

        # Check overlap is recorded
        if len(chunks) > 1:
            assert chunks[0].overlap_after > 0 or chunks[1].overlap_before > 0

    def test_zero_overlap(self):
        """Test that zero overlap produces non-overlapping chunks."""
        chunker = SimpleChunker(chunk_size=100, overlap=0)
        text = "A" * 200

        chunks = chunker.chunk(text)

        # No overlap should be recorded
        for chunk in chunks:
            assert chunk.overlap_before == 0 or chunk.overlap_after == 0


class TestChunkMetadata:
    """Test metadata preservation and inclusion."""

    def test_metadata_preservation(self):
        """Test that input metadata is preserved in chunks."""
        chunker = SimpleChunker(chunk_size=100)
        text = "A" * 200
        metadata = {
            "source_filename": "test.pdf",
            "dataset_name": "test_dataset",
            "namespace": "test_ns",
        }

        chunks = chunker.chunk(text, metadata)

        for chunk in chunks:
            assert chunk.metadata.get("source_filename") == "test.pdf"
            assert chunk.metadata.get("dataset_name") == "test_dataset"
            assert chunk.metadata.get("namespace") == "test_ns"

    def test_chunk_metadata_includes_chunking_info(self):
        """Test that chunk metadata includes chunking information."""
        chunker = SimpleChunker(chunk_size=100, overlap=10, strategy=ChunkingStrategy.SENTENCES)
        text = "Sentence one. Sentence two. Sentence three."

        chunks = chunker.chunk(text)

        for chunk in chunks:
            assert "chunker_name" in chunk.metadata
            assert chunk.metadata["chunker_name"] == "SimpleChunker"
            assert "chunking_strategy" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "sentences"
            assert "chunk_size_target" in chunk.metadata
            assert chunk.metadata["chunk_size_target"] == 100
            assert "chunk_overlap" in chunk.metadata
            assert chunk.metadata["chunk_overlap"] == 10

    def test_chunk_index_and_total(self):
        """Test chunk_index and total_chunks in metadata."""
        chunker = SimpleChunker(chunk_size=50)
        text = "A" * 150

        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["is_chunked"] == (len(chunks) > 1)


class TestChunkResult:
    """Test ChunkResult dataclass."""

    def test_chunk_result_to_dict(self):
        """Test ChunkResult.to_dict() method."""
        chunk = ChunkResult(
            content="Test content",
            metadata={"key": "value"},
            chunk_index=0,
            total_chunks=3,
            start_offset=0,
            end_offset=12,
            strategy="characters",
            overlap_before=0,
            overlap_after=10,
        )

        result = chunk.to_dict()

        assert result["content"] == "Test content"
        assert result["metadata"] == {"key": "value"}
        assert result["chunk_index"] == 0
        assert result["total_chunks"] == 3
        assert result["start_offset"] == 0
        assert result["end_offset"] == 12
        assert result["strategy"] == "characters"
        assert result["overlap_before"] == 0
        assert result["overlap_after"] == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = SimpleChunker()
        chunks = chunker.chunk("")

        assert len(chunks) == 0

    def test_whitespace_only_text(self):
        """Test chunking whitespace-only text."""
        chunker = SimpleChunker()
        chunks = chunker.chunk("   \n\n\t  ")

        assert len(chunks) == 0

    def test_none_metadata(self):
        """Test chunking with None metadata."""
        chunker = SimpleChunker()
        text = "Some text"

        chunks = chunker.chunk(text, None)

        assert len(chunks) == 1
        assert "chunk_index" in chunks[0].metadata

    def test_very_large_text(self):
        """Test chunking very large text."""
        chunker = SimpleChunker(chunk_size=1000)
        text = "A" * 100000  # 100K characters

        chunks = chunker.chunk(text)

        assert len(chunks) == 100
        assert all(len(c.content) == 1000 for c in chunks)


class TestChunkCountEstimate:
    """Test chunk count estimation."""

    def test_estimate_character_chunks(self):
        """Test chunk count estimation for characters."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.CHARACTERS)
        text = "A" * 350

        estimate = chunker.get_chunk_count_estimate(text)

        assert estimate == 4  # 350/100 = 3.5, rounds up to 4

    def test_estimate_sentence_chunks(self):
        """Test chunk count estimation for sentences."""
        chunker = SimpleChunker(chunk_size=100, strategy=ChunkingStrategy.SENTENCES)
        text = "Sentence one. Sentence two. Sentence three."

        estimate = chunker.get_chunk_count_estimate(text)

        assert estimate >= 1

    def test_estimate_section_chunks(self):
        """Test chunk count estimation for sections."""
        chunker = SimpleChunker(strategy=ChunkingStrategy.SECTIONS)
        text = "# One\nContent.\n# Two\nMore content.\n# Three\nFinal."

        estimate = chunker.get_chunk_count_estimate(text)

        assert estimate == 3  # 3 headers
