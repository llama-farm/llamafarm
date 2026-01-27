"""Tests for the UniversalParser."""

import tempfile
from pathlib import Path
from unittest.mock import patch


class TestUniversalParserImport:
    """Test UniversalParser import and initialization."""

    def test_universal_parser_can_be_imported(self):
        """Test: UniversalParser can be imported without errors."""
        from components.parsers.universal import UniversalParser

        assert UniversalParser is not None

    def test_universal_parser_initialization(self):
        """Test: UniversalParser initializes with default config."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()
        assert parser.chunk_size == 1024
        assert parser.chunk_overlap == 100
        assert parser.chunk_strategy == "semantic"
        assert parser.use_ocr is True

    def test_universal_parser_custom_config(self):
        """Test: UniversalParser accepts custom configuration."""
        from components.parsers.universal import UniversalParser

        config = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "chunk_strategy": "paragraphs",
            "use_ocr": False,
        }
        parser = UniversalParser(config=config)
        assert parser.chunk_size == 512
        assert parser.chunk_overlap == 50
        assert parser.chunk_strategy == "paragraphs"
        assert parser.use_ocr is False


class TestUniversalParserGracefulDegradation:
    """Test graceful degradation when dependencies are missing."""

    def test_handles_missing_markitdown_gracefully(self):
        """Test: UniversalParser handles missing MarkItDown gracefully."""
        with patch.dict("sys.modules", {"markitdown": None}):
            # Force reimport with mocked missing module
            import importlib

            import components.parsers.universal.universal_parser as parser_module

            # Save original values
            original_available = parser_module.MARKITDOWN_AVAILABLE

            try:
                # Simulate missing dependency
                parser_module.MARKITDOWN_AVAILABLE = False
                parser_module.MarkItDown = None

                from components.parsers.universal import UniversalParser

                # Should still initialize without error
                parser = UniversalParser()
                assert parser is not None
                assert parser._markitdown is None

            finally:
                # Restore
                parser_module.MARKITDOWN_AVAILABLE = original_available
                importlib.reload(parser_module)

    def test_handles_missing_semchunk_gracefully(self):
        """Test: UniversalParser handles missing semchunk gracefully."""
        import components.parsers.universal.universal_parser as parser_module

        original_available = parser_module.SEMCHUNK_AVAILABLE

        try:
            parser_module.SEMCHUNK_AVAILABLE = False

            from components.parsers.universal import UniversalParser

            parser = UniversalParser(config={"chunk_strategy": "semantic"})
            assert parser is not None

            # Should fall back to another strategy when chunking
            text = "This is a test. " * 100
            chunks = parser._chunk_text(text)
            assert len(chunks) >= 1

        finally:
            parser_module.SEMCHUNK_AVAILABLE = original_available


class TestUniversalParserParsing:
    """Test parsing functionality."""

    def test_parses_plain_text_produces_documents(self):
        """Test: UniversalParser parses plain text and produces Documents."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "paragraphs"})

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write("This is paragraph one with enough content to meet minimum.\n\n")
            tmp.write("This is paragraph two with sufficient content for parsing.\n\n")
            tmp.write("This is paragraph three with adequate text for testing.")
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) >= 1
            assert result.documents[0].content is not None
            assert len(result.documents[0].content) > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_returns_empty_for_nonexistent_file(self):
        """Test: UniversalParser returns error for non-existent file."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()
        result = parser.parse("/nonexistent/path/file.txt")
        assert len(result.documents) == 0
        assert len(result.errors) > 0


class TestUniversalParserChunking:
    """Test chunking functionality."""

    def test_chunks_text_respecting_chunk_size(self):
        """Test: UniversalParser chunks text respecting chunk_size config."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_size": 200,
            "chunk_strategy": "characters",
            "min_chunk_size": 20,
        })

        # Create a long text
        text = "This is a test sentence with some words. " * 50

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            # Should produce multiple chunks
            assert len(result.documents) > 1
            # Each chunk should be around chunk_size (with some tolerance)
            for doc in result.documents[:-1]:  # Last chunk may be smaller
                assert len(doc.content) <= 250  # Some tolerance
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_supports_semantic_chunk_strategy(self):
        """Test: UniversalParser supports semantic chunk strategy."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "semantic"})
        assert parser.chunk_strategy == "semantic"

    def test_supports_sections_chunk_strategy(self):
        """Test: UniversalParser supports sections chunk strategy."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "sections"})

        # Create markdown with headers
        markdown_text = """# Introduction
This is the introduction section with enough content.

## Section One
Content for section one with sufficient text here.

## Section Two
Content for section two with adequate content here.

### Subsection
Content for subsection with proper length text.
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp:
            tmp.write(markdown_text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            # Should produce multiple chunks based on sections
            assert len(result.documents) >= 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_supports_paragraphs_chunk_strategy(self):
        """Test: UniversalParser supports paragraphs chunk strategy."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "paragraphs"})

        text = """First paragraph with sufficient content for testing purposes.

Second paragraph with enough text to be recognized as separate content.

Third paragraph with adequate length for the chunking test."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) >= 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_supports_sentences_chunk_strategy(self):
        """Test: UniversalParser supports sentences chunk strategy."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_strategy": "sentences",
            "chunk_size": 100,
            "min_chunk_size": 20,
        })

        text = (
            "This is the first sentence. "
            "This is the second sentence with more content. "
            "Here comes the third sentence which is also important. "
            "And finally the fourth sentence to complete the text."
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) >= 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_supports_characters_chunk_strategy(self):
        """Test: UniversalParser supports characters chunk strategy."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_strategy": "characters",
            "chunk_size": 100,
            "min_chunk_size": 20,
        })

        text = "A" * 500  # Long text to be chunked

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) > 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestUniversalParserMetadata:
    """Test chunk metadata generation."""

    def test_chunk_has_chunk_index_metadata(self):
        """Test: Each chunk has correct chunk_index metadata."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_strategy": "characters",
            "chunk_size": 100,
            "min_chunk_size": 20,
        })

        text = "Test content. " * 50

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            for i, doc in enumerate(result.documents):
                assert "chunk_index" in doc.metadata
                assert doc.metadata["chunk_index"] == i
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_chunk_has_chunk_label_metadata(self):
        """Test: Each chunk has chunk_label metadata in N/M format."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_strategy": "characters",
            "chunk_size": 100,
            "min_chunk_size": 20,
        })

        text = "Test content. " * 50

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            total = len(result.documents)
            for i, doc in enumerate(result.documents):
                assert "chunk_label" in doc.metadata
                assert doc.metadata["chunk_label"] == f"{i + 1}/{total}"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_chunk_has_total_chunks_metadata(self):
        """Test: Each chunk has total_chunks metadata."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={
            "chunk_strategy": "characters",
            "chunk_size": 100,
            "min_chunk_size": 20,
        })

        text = "Test content. " * 50

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            total = len(result.documents)
            for doc in result.documents:
                assert "total_chunks" in doc.metadata
                assert doc.metadata["total_chunks"] == total
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_chunk_has_document_name_metadata(self):
        """Test: Each chunk has document_name metadata."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "paragraphs"})

        text = "Test content with enough text to meet minimum chunk size requirements."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
            filename = Path(tmp_path).name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) >= 1
            assert "document_name" in result.documents[0].metadata
            assert result.documents[0].metadata["document_name"] == filename
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_chunk_has_word_count_metadata(self):
        """Test: Each chunk has word_count metadata."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "paragraphs"})

        text = "This is a test sentence with exactly eleven words in it."

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        try:
            result = parser.parse(tmp_path)
            assert len(result.documents) >= 1
            assert "word_count" in result.documents[0].metadata
            assert result.documents[0].metadata["word_count"] > 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestUniversalParserCanParse:
    """Test can_parse functionality."""

    def test_can_parse_supported_extensions(self):
        """Test: can_parse returns True for supported extensions."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        supported = [".txt", ".md", ".pdf", ".docx", ".html", ".csv", ".json"]
        for ext in supported:
            assert parser.can_parse(f"test{ext}"), f"Should support {ext}"

    def test_can_parse_unsupported_extensions(self):
        """Test: can_parse returns False for unsupported extensions."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        unsupported = [".xyz", ".abc", ".unknown"]
        for ext in unsupported:
            assert not parser.can_parse(f"test{ext}"), f"Should not support {ext}"


class TestUniversalParserParseBlob:
    """Test parse_blob functionality."""

    def test_parse_blob_processes_bytes(self):
        """Test: parse_blob processes raw bytes correctly."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser(config={"chunk_strategy": "paragraphs"})

        content = b"This is test content from blob data with sufficient length."
        metadata = {"filename": "test.txt", "source": "blob"}

        documents = parser.parse_blob(content, metadata)
        assert len(documents) >= 1
        assert "blob" in documents[0].metadata.get("source", "")
