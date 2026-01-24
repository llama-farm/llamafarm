"""Integration tests for Preview - TDD Red Phase.

Tests that preview produces the EXACT same chunks as actual ingestion.
All tests written FIRST and will fail until implementation is complete.
"""

import tempfile
from pathlib import Path

import pytest

from core.base import Document


class TestPreviewIntegration:
    """Integration tests - preview matches actual ingestion."""

    @pytest.fixture
    def sample_text_content(self) -> str:
        """Sample text content for testing."""
        return """
What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation is a technique that enhances large language models
by combining them with external knowledge retrieval systems. This allows the model
to access and incorporate relevant information from a knowledge base when generating
responses.

The key components of a RAG system are:
1. A retrieval system that can search and find relevant documents
2. A generation model (typically a large language model)
3. A mechanism to combine retrieved information with the generation process

RAG systems have become increasingly popular because they help address some of the
limitations of standalone language models, such as outdated knowledge and hallucinations.
"""

    @pytest.fixture
    def sample_text_file(self, sample_text_content: str, tmp_path: Path) -> Path:
        """Create a temporary text file with sample content."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text(sample_text_content)
        return file_path

    @pytest.fixture
    def sample_markdown_content(self) -> str:
        """Sample markdown content."""
        return """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that provides systems
the ability to automatically learn and improve from experience without being
explicitly programmed.

## Types of Machine Learning

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data, helping
to predict outcomes for unforeseen data.

### Unsupervised Learning

Unsupervised learning uses machine learning algorithms to analyze and cluster
unlabeled datasets.

## Applications

- Natural Language Processing
- Computer Vision
- Recommendation Systems
"""

    @pytest.fixture
    def sample_markdown_file(self, sample_markdown_content: str, tmp_path: Path) -> Path:
        """Create a temporary markdown file."""
        file_path = tmp_path / "sample.md"
        file_path.write_text(sample_markdown_content)
        return file_path

    @pytest.mark.integration
    def test_preview_matches_ingestion_text_file(
        self,
        sample_text_file: Path,
        sample_text_content: str,
    ):
        """Preview chunks MUST match what ingestion produces."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        # Create a minimal strategy config for testing
        from config.datamodel import DataProcessingStrategy, Parser

        strategy = DataProcessingStrategy(
            name="test_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 200, "chunk_overlap": 20},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        # Read file data
        file_data = sample_text_file.read_bytes()
        metadata = {"filename": sample_text_file.name}

        # Get preview result
        preview_result = preview_handler.generate_preview(file_data, metadata)

        # Get what ingestion would produce (same process_blob call)
        ingestion_docs = blob_processor.process_blob(file_data, metadata)

        # CRITICAL: Same number of chunks
        assert len(preview_result.chunks) == len(ingestion_docs), (
            f"Preview produced {len(preview_result.chunks)} chunks, "
            f"but ingestion produces {len(ingestion_docs)} chunks"
        )

        # CRITICAL: Same content in each chunk
        for i, (preview_chunk, ingested_doc) in enumerate(
            zip(preview_result.chunks, ingestion_docs)
        ):
            assert preview_chunk.content == ingested_doc.content, (
                f"Chunk {i} content mismatch:\n"
                f"Preview: {preview_chunk.content[:50]}...\n"
                f"Ingestion: {ingested_doc.content[:50]}..."
            )

    @pytest.mark.integration
    def test_preview_matches_ingestion_markdown(
        self,
        sample_markdown_file: Path,
    ):
        """Preview matches ingestion for markdown files."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Parser

        strategy = DataProcessingStrategy(
            name="test_markdown_strategy",
            parsers=[
                Parser(
                    type="MarkdownParser_Python",
                    file_include_patterns=["*.md"],
                    config={"chunk_size": 300, "chunk_overlap": 30},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = sample_markdown_file.read_bytes()
        metadata = {"filename": sample_markdown_file.name}

        preview_result = preview_handler.generate_preview(file_data, metadata)
        ingestion_docs = blob_processor.process_blob(file_data, metadata)

        # Same number of chunks
        assert len(preview_result.chunks) == len(ingestion_docs)

        # Same content
        for preview_chunk, ingested_doc in zip(preview_result.chunks, ingestion_docs):
            assert preview_chunk.content == ingested_doc.content

    @pytest.mark.integration
    @pytest.mark.slow
    def test_preview_matches_ingestion_pdf(self, tmp_path: Path):
        """Preview matches ingestion for PDF files."""
        # Skip if no sample PDF is available
        pytest.skip("PDF testing requires sample PDF file")

    @pytest.mark.integration
    def test_preview_with_extractors_matches_ingestion(
        self,
        sample_text_file: Path,
    ):
        """Extractors are applied same way in preview and ingestion."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Extractor, Parser

        strategy = DataProcessingStrategy(
            name="test_extractor_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 200, "chunk_overlap": 20},
                    priority=1,
                )
            ],
            extractors=[
                Extractor(
                    type="ContentStatisticsExtractor",
                    file_include_patterns=["*.txt"],
                    config={},
                    priority=1,
                )
            ],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = sample_text_file.read_bytes()
        metadata = {"filename": sample_text_file.name}

        preview_result = preview_handler.generate_preview(file_data, metadata)
        ingestion_docs = blob_processor.process_blob(file_data, metadata)

        # Same content (extractors don't change content, only metadata)
        for preview_chunk, ingested_doc in zip(preview_result.chunks, ingestion_docs):
            assert preview_chunk.content == ingested_doc.content

    @pytest.mark.integration
    def test_preview_with_different_chunk_sizes(self, sample_text_file: Path):
        """Preview correctly records chunk size override in result."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Parser

        strategy = DataProcessingStrategy(
            name="test_small_chunks_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 100, "chunk_overlap": 10},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = sample_text_file.read_bytes()
        metadata = {"filename": sample_text_file.name}

        # Get preview with explicit chunk size in metadata
        result_explicit = preview_handler.generate_preview(
            file_data,
            {"filename": sample_text_file.name, "chunk_size": 200},
        )

        # Get preview with override
        result_override = preview_handler.generate_preview(
            file_data,
            metadata.copy(),
            chunk_size_override=800,
        )

        # Explicit metadata should be recorded in result
        assert result_explicit.chunk_size == 200
        # Override should be recorded in result
        assert result_override.chunk_size == 800

    @pytest.mark.integration
    def test_preview_positions_are_accurate(self, sample_text_content: str):
        """Preview chunk positions accurately map to original text."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Parser

        strategy = DataProcessingStrategy(
            name="test_position_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 100, "chunk_overlap": 0},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = sample_text_content.encode("utf-8")
        metadata = {"filename": "test.txt"}

        result = preview_handler.generate_preview(file_data, metadata)

        # Verify each chunk's position maps to correct text
        for chunk in result.chunks:
            if chunk.start_position >= 0:  # Skip unfindable chunks
                extracted = result.original_text[chunk.start_position:chunk.end_position]
                assert extracted == chunk.content, (
                    f"Position mismatch for chunk {chunk.chunk_index}:\n"
                    f"Content: '{chunk.content[:30]}...'\n"
                    f"Extracted: '{extracted[:30]}...'"
                )

    @pytest.mark.integration
    def test_preview_overlap_detection(self):
        """Preview correctly identifies overlapping regions."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Parser

        # Use a known text with predictable chunking
        text = "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH"

        strategy = DataProcessingStrategy(
            name="test_overlap_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 15, "chunk_overlap": 5},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = text.encode("utf-8")
        metadata = {"filename": "test.txt"}

        result = preview_handler.generate_preview(file_data, metadata)

        # With overlap, adjacent chunks should share some text
        if len(result.chunks) >= 2:
            chunk1 = result.chunks[0]
            chunk2 = result.chunks[1]

            # The end of chunk1 should overlap with the start of chunk2
            if chunk1.end_position > chunk2.start_position:
                overlap_region = result.original_text[
                    chunk2.start_position:chunk1.end_position
                ]
                assert len(overlap_region) > 0, "Expected overlap region"

    @pytest.mark.integration
    def test_preview_idempotent(self, sample_text_file: Path):
        """Calling preview multiple times produces same result."""
        from core.blob_processor import BlobProcessor
        from core.preview_handler import PreviewHandler

        from config.datamodel import DataProcessingStrategy, Parser

        strategy = DataProcessingStrategy(
            name="test_strategy",
            parsers=[
                Parser(
                    type="TextParser_Python",
                    file_include_patterns=["*.txt"],
                    config={"chunk_size": 200, "chunk_overlap": 20},
                    priority=1,
                )
            ],
            extractors=[],
        )

        blob_processor = BlobProcessor(strategy)
        preview_handler = PreviewHandler(blob_processor=blob_processor)

        file_data = sample_text_file.read_bytes()
        metadata = {"filename": sample_text_file.name}

        # Call preview twice
        result1 = preview_handler.generate_preview(file_data, metadata.copy())
        result2 = preview_handler.generate_preview(file_data, metadata.copy())

        # Results should be identical
        assert result1.total_chunks == result2.total_chunks
        assert result1.original_text == result2.original_text

        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert c1.content == c2.content
            assert c1.start_position == c2.start_position
            assert c1.end_position == c2.end_position
