"""Tests for the UniversalExtractor."""

# Import from the rag package
import sys
import tempfile
from pathlib import Path

import pytest

rag_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(rag_dir))

from core.base import Document  # noqa: E402


class TestUniversalExtractorImport:
    """Test UniversalExtractor import and initialization."""

    def test_universal_extractor_can_be_imported(self):
        """Test: UniversalExtractor can be imported without errors."""
        from components.extractors.universal_extractor import UniversalExtractor

        assert UniversalExtractor is not None

    def test_universal_extractor_initialization(self):
        """Test: UniversalExtractor initializes with default config."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()
        assert extractor.keyword_count == 10
        assert extractor.generate_summary is True
        assert extractor.detect_language is True

    def test_universal_extractor_custom_config(self):
        """Test: UniversalExtractor accepts custom configuration."""
        from components.extractors.universal_extractor import UniversalExtractor

        config = {
            "keyword_count": 5,
            "generate_summary": False,
            "summary_sentences": 2,
        }
        extractor = UniversalExtractor(config=config)
        assert extractor.keyword_count == 5
        assert extractor.generate_summary is False
        assert extractor.summary_sentences == 2


class TestUniversalExtractorGracefulDegradation:
    """Test graceful degradation when dependencies are missing."""

    def test_handles_missing_yake_gracefully(self):
        """Test: UniversalExtractor handles missing YAKE gracefully."""
        import components.extractors.universal_extractor.universal_extractor as extractor_module

        original_available = extractor_module.YAKE_AVAILABLE

        try:
            extractor_module.YAKE_AVAILABLE = False

            from components.extractors.universal_extractor import UniversalExtractor

            extractor = UniversalExtractor()
            assert extractor is not None

            # Should not crash when extracting
            doc = Document(content="Test content for extraction.", metadata={})
            result = extractor.extract([doc])
            assert len(result) == 1

        finally:
            extractor_module.YAKE_AVAILABLE = original_available


class TestUniversalExtractorKeywords:
    """Test keyword extraction functionality."""

    def test_extracts_keywords_from_text(self):
        """Test: UniversalExtractor extracts keywords from text."""
        import components.extractors.universal_extractor.universal_extractor as mod
        from components.extractors.universal_extractor import UniversalExtractor

        if not mod.YAKE_AVAILABLE:
            pytest.skip("YAKE not available")

        extractor = UniversalExtractor(config={"keyword_count": 5})

        doc = Document(
            content="""
            Machine learning is a subset of artificial intelligence.
            Deep learning uses neural networks for pattern recognition.
            Natural language processing enables computers to understand text.
            """,
            metadata={},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "keywords" in result[0].metadata
        # Should have extracted some keywords
        assert len(result[0].metadata["keywords"]) > 0


class TestUniversalExtractorDocumentMetadata:
    """Test document-level metadata extraction."""

    def test_adds_document_name_metadata(self):
        """Test: UniversalExtractor adds document-level metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write("Test content for metadata extraction testing purposes.")
            tmp_path = tmp.name
            filename = Path(tmp_path).name

        try:
            doc = Document(
                content="Test content for metadata extraction testing purposes.",
                metadata={},
                source=tmp_path,
            )

            result = extractor.extract([doc])
            assert len(result) == 1
            assert "document_name" in result[0].metadata
            assert result[0].metadata["document_name"] == filename
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_adds_document_type_metadata(self):
        """Test: UniversalExtractor adds document_type metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write("# Test markdown content with headers and text.")
            tmp_path = tmp.name

        try:
            doc = Document(
                content="# Test markdown content with headers and text.",
                metadata={},
                source=tmp_path,
            )

            result = extractor.extract([doc])
            assert len(result) == 1
            assert "document_type" in result[0].metadata
            assert result[0].metadata["document_type"] == ".md"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_adds_document_size_metadata(self):
        """Test: UniversalExtractor adds document_size metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        content = "Test content with known size for testing document size metadata."
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            doc = Document(content=content, metadata={}, source=tmp_path)

            result = extractor.extract([doc])
            assert len(result) == 1
            assert "document_size" in result[0].metadata
            assert result[0].metadata["document_size"] > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestUniversalExtractorChunkMetadata:
    """Test chunk-level metadata extraction."""

    def test_adds_chunk_label_metadata(self):
        """Test: UniversalExtractor adds chunk_label metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        doc = Document(
            content="Test content for chunk label extraction.",
            metadata={"chunk_index": 2, "total_chunks": 5},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "chunk_label" in result[0].metadata
        assert result[0].metadata["chunk_label"] == "3/5"

    def test_adds_chunk_position_metadata(self):
        """Test: UniversalExtractor adds chunk_position metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        # Test start position
        doc_start = Document(
            content="First chunk content.",
            metadata={"chunk_index": 0, "total_chunks": 3},
        )
        result = extractor.extract([doc_start])
        assert result[0].metadata["chunk_position"] == "start"

        # Test middle position
        doc_middle = Document(
            content="Middle chunk content.",
            metadata={"chunk_index": 1, "total_chunks": 3},
        )
        result = extractor.extract([doc_middle])
        assert result[0].metadata["chunk_position"] == "middle"

        # Test end position
        doc_end = Document(
            content="Last chunk content.",
            metadata={"chunk_index": 2, "total_chunks": 3},
        )
        result = extractor.extract([doc_end])
        assert result[0].metadata["chunk_position"] == "end"

        # Test only position
        doc_only = Document(
            content="Only chunk content.",
            metadata={"chunk_index": 0, "total_chunks": 1},
        )
        result = extractor.extract([doc_only])
        assert result[0].metadata["chunk_position"] == "only"

    def test_adds_word_count_metadata(self):
        """Test: UniversalExtractor adds word_count metadata."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        doc = Document(
            content="This is exactly seven words in total.",
            metadata={},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "word_count" in result[0].metadata
        assert result[0].metadata["word_count"] == 7


class TestUniversalExtractorAuthorTitle:
    """Test author/title extraction from document metadata."""

    def test_preserves_author_metadata(self):
        """Test: UniversalExtractor preserves author metadata if present."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        doc = Document(
            content="Document with author information.",
            metadata={"author": "John Doe"},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "author" in result[0].metadata
        assert result[0].metadata["author"] == "John Doe"

    def test_preserves_title_metadata(self):
        """Test: UniversalExtractor preserves title metadata if present."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        doc = Document(
            content="Document with title information.",
            metadata={"title": "My Document Title"},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "title" in result[0].metadata
        assert result[0].metadata["title"] == "My Document Title"


class TestUniversalExtractorTimestamps:
    """Test timestamp metadata."""

    def test_adds_processing_timestamps(self):
        """Test: UniversalExtractor adds processing timestamps."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        doc = Document(
            content="Document content for timestamp testing.",
            metadata={},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "processed_at" in result[0].metadata
        # Should be ISO format timestamp
        timestamp = result[0].metadata["processed_at"]
        assert "T" in timestamp  # ISO format indicator


class TestUniversalExtractorSummary:
    """Test summary generation."""

    def test_generates_summary_from_first_sentences(self):
        """Test: UniversalExtractor generates summary from first 3 sentences."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor(config={"summary_sentences": 2})

        doc = Document(
            content=(
                "First sentence of the document. "
                "Second sentence with more info. "
                "Third sentence that should not appear in summary."
            ),
            metadata={},
        )

        result = extractor.extract([doc])
        assert len(result) == 1
        assert "summary" in result[0].metadata
        summary = result[0].metadata["summary"]
        assert "First sentence" in summary
        # Summary should be limited to configured sentences
        assert len(summary) < len(doc.content)


class TestUniversalExtractorTableCodeDetection:
    """Test table and code block detection."""

    def test_detects_tables(self):
        """Test: UniversalExtractor detects tables."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        # Markdown table
        doc_table = Document(
            content="""
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
            """,
            metadata={},
        )

        result = extractor.extract([doc_table])
        assert result[0].metadata.get("has_tables") is True

        # No table
        doc_no_table = Document(
            content="Just regular text without any tables.",
            metadata={},
        )

        result = extractor.extract([doc_no_table])
        assert result[0].metadata.get("has_tables") is False

    def test_detects_code_blocks(self):
        """Test: UniversalExtractor detects code blocks."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()

        # Markdown code fence
        doc_code = Document(
            content="""
```python
def hello():
    print("Hello")
```
            """,
            metadata={},
        )

        result = extractor.extract([doc_code])
        assert result[0].metadata.get("has_code") is True

        # No code
        doc_no_code = Document(
            content="Just regular text without any code blocks.",
            metadata={},
        )

        result = extractor.extract([doc_no_code])
        assert result[0].metadata.get("has_code") is False


class TestUniversalExtractorGetDependencies:
    """Test get_dependencies method."""

    def test_get_dependencies_returns_empty_list(self):
        """Test: get_dependencies returns empty list (all optional)."""
        from components.extractors.universal_extractor import UniversalExtractor

        extractor = UniversalExtractor()
        deps = extractor.get_dependencies()
        assert isinstance(deps, list)
        # All dependencies are optional for graceful degradation
        assert len(deps) == 0
