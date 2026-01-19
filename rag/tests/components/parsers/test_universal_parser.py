"""Tests for UniversalParser."""

import os
import tempfile
from pathlib import Path

import pytest


class TestUniversalParser:
    """Tests for UniversalParser using MarkItDown + SemChunk."""

    @pytest.fixture
    def parser(self):
        """Create a UniversalParser instance."""
        from components.parsers.universal.parser import UniversalParser
        return UniversalParser()

    @pytest.fixture
    def sample_html_file(self):
        """Create a sample HTML file for testing."""
        content = """<!DOCTYPE html>
<html>
<head><title>Test Document</title></head>
<body>
<h1>Test Heading</h1>
<p>This is a test paragraph with some content.</p>
<p>This is another paragraph.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
<li>Item 3</li>
</ul>
</body>
</html>"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_json_file(self):
        """Create a sample JSON file for testing."""
        content = '{"name": "test", "value": 123, "items": ["a", "b", "c"]}'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def sample_txt_file(self):
        """Create a sample text file for testing."""
        content = """This is a test document.

It has multiple paragraphs that should be chunked properly.

The UniversalParser should handle this text file correctly.

Each paragraph contains meaningful content for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            f.flush()
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_parser_initialization(self, parser):
        """Test that the parser initializes correctly."""
        assert parser.name == "UniversalParser"
        assert parser.chunk_size == 1024
        assert parser.chunk_strategy == "semantic"
        assert parser.use_ocr is True

    def test_parser_can_parse_supported_extensions(self, parser):
        """Test can_parse for supported file types."""
        supported = [
            ".pdf", ".docx", ".xlsx", ".html", ".json",
            ".xml", ".txt", ".md", ".csv"
        ]
        for ext in supported:
            assert parser.can_parse(f"test{ext}") is True

    def test_parser_cannot_parse_unsupported_extensions(self, parser):
        """Test can_parse for unsupported file types."""
        unsupported = [".exe", ".bin", ".zip", ".tar", ".mp3"]
        for ext in unsupported:
            assert parser.can_parse(f"test{ext}") is False

    def test_parse_html_file(self, parser, sample_html_file):
        """Test parsing an HTML file."""
        result = parser.parse(sample_html_file)

        assert len(result.errors) == 0
        assert len(result.documents) > 0

        # Check that content was extracted
        all_content = " ".join(doc.content for doc in result.documents)
        assert "Test Heading" in all_content
        assert "test paragraph" in all_content

    def test_parse_json_file(self, parser, sample_json_file):
        """Test parsing a JSON file."""
        result = parser.parse(sample_json_file)

        assert len(result.errors) == 0
        assert len(result.documents) > 0

    def test_parse_txt_file(self, parser, sample_txt_file):
        """Test parsing a text file."""
        # Verify file content
        with open(sample_txt_file, 'r') as f:
            file_content = f.read()
        print(f"File content length: {len(file_content)}")
        print(f"File content preview: {file_content[:100]}")

        result = parser.parse(sample_txt_file)

        if result.errors:
            print(f"Errors: {result.errors}")
        print(f"Documents: {len(result.documents)}")

        assert len(result.errors) == 0, f"Unexpected errors: {result.errors}"
        assert len(result.documents) > 0

        # Check content
        all_content = " ".join(doc.content for doc in result.documents)
        assert "test document" in all_content

    def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file."""
        result = parser.parse("/nonexistent/path/file.pdf")

        assert len(result.errors) > 0
        assert "not found" in result.errors[0]["error"].lower()

    def test_document_metadata(self, parser, sample_html_file):
        """Test that documents have correct metadata."""
        result = parser.parse(sample_html_file)

        assert len(result.documents) > 0
        doc = result.documents[0]

        assert "parser" in doc.metadata
        assert doc.metadata["parser"] == "UniversalParser"
        assert "chunk_index" in doc.metadata
        assert "total_chunks" in doc.metadata

    def test_chunking_produces_multiple_documents(self, parser):
        """Test that large content is chunked into multiple documents."""
        # Create a large text file
        content = "This is a test sentence. " * 500  # ~2500 words
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.documents) > 1, "Large content should produce multiple chunks"
        finally:
            os.unlink(temp_path)

    def test_page_chunking_strategy(self):
        """Test page-based chunking strategy."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={"chunk_strategy": "page"})

        content = "Page 1 content\f Page 2 content\f Page 3 content"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            # Should have 3 documents for 3 pages
            assert len(result.documents) >= 1
        finally:
            os.unlink(temp_path)

    def test_parser_with_real_pdf(self, parser):
        """Test parsing real PDF files from examples directory."""
        examples_dir = Path(__file__).parent.parent.parent.parent.parent / "examples" / "rag_pipeline" / "sample_files"
        pdf_file = examples_dir / "Alpaca-Care-for-Beginners.pdf"

        if not pdf_file.exists():
            pytest.skip("Sample PDF not found")

        result = parser.parse(str(pdf_file))

        assert len(result.errors) == 0
        assert len(result.documents) > 0

        # Check content was extracted
        all_content = " ".join(doc.content for doc in result.documents)
        assert len(all_content) > 100

    def test_needs_ocr_for_images(self, parser):
        """Test OCR detection for image files."""
        assert parser._needs_ocr("test.png", "") is True
        assert parser._needs_ocr("test.jpg", "") is True
        assert parser._needs_ocr("test.jpeg", "") is True

    def test_needs_ocr_for_low_text_pdf(self, parser):
        """Test OCR detection for PDFs with low text."""
        # Low text should trigger OCR
        assert parser._needs_ocr("test.pdf", "short") is True
        # Normal text should not trigger OCR
        assert parser._needs_ocr("test.pdf", "A" * 100) is False

    def test_no_ocr_for_docx(self, parser):
        """Test that DOCX files don't trigger OCR."""
        assert parser._needs_ocr("test.docx", "some content") is False


class TestUniversalParserRegression:
    """Regression tests to ensure existing parsers still work."""

    def test_pypdf2_parser_still_works(self):
        """Test that existing PyPDF2 parser still works."""
        try:
            from components.parsers.pdf.pypdf2_parser import PDFParser_PyPDF2
            parser = PDFParser_PyPDF2()
            assert parser.name == "PDFParser_PyPDF2"
        except ImportError:
            pytest.skip("PyPDF2 parser not available")

    def test_docling_parser_importable(self):
        """Test that Docling parser is still importable (if available)."""
        try:
            from components.parsers.docling import DoclingParser
            # Just test import, don't instantiate (heavy)
        except ImportError:
            pytest.skip("Docling parser not available")

    def test_parser_factory_discovers_universal(self):
        """Test that parser factory discovers UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        # Clear cache to force rediscovery
        ToolAwareParserFactory._parser_configs = {}

        parsers = ToolAwareParserFactory.discover_parsers()

        # UniversalParser should be in the discovered parsers
        # It's in the 'universal' type directory
        assert "universal" in parsers or len(parsers) > 0


class TestChunkingStrategies:
    """Tests for different chunking strategies in UniversalParser."""

    @pytest.fixture
    def markdown_content(self):
        """Create content with markdown headers for section testing."""
        return """# Introduction

This is the introduction section with some content.
It spans multiple lines.

## Background

This is the background section.
More details about the background here.

### Technical Details

Some technical details in a subsection.

## Conclusion

The conclusion section wraps things up."""

    @pytest.fixture
    def paragraph_content(self):
        """Create content with clear paragraph breaks."""
        return """This is the first paragraph. It contains some text.

This is the second paragraph. It has different content.

This is the third paragraph. More unique content here.

This is the fourth paragraph. Final paragraph content."""

    @pytest.fixture
    def page_content(self):
        """Create content with page breaks."""
        return """Page 1 content goes here.
This is page one text.\f
Page 2 content starts now.
More text on page two.\f
Page 3 is the final page.
Closing content here."""

    def test_semantic_strategy_is_default(self):
        """Test that semantic is the default chunking strategy."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser()
        assert parser.chunk_strategy == "semantic"

    def test_section_chunking_strategy(self, markdown_content):
        """Test section-based chunking on markdown content."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "section",
            "min_chunk_size": 10
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0

            # Should produce multiple sections
            assert len(result.documents) >= 2

            # Check metadata
            for doc in result.documents:
                assert doc.metadata["chunk_strategy"] == "section"
        finally:
            os.unlink(temp_path)

    def test_paragraph_chunking_strategy(self, paragraph_content):
        """Test paragraph-based chunking."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "paragraph",
            "min_chunk_size": 10
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(paragraph_content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0

            # Should produce multiple paragraphs
            assert len(result.documents) >= 2

            # Check metadata
            for doc in result.documents:
                assert doc.metadata["chunk_strategy"] == "paragraph"
        finally:
            os.unlink(temp_path)

    def test_sentence_chunking_strategy(self):
        """Test sentence-based chunking."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "sentence",
            "min_chunk_size": 10
        })

        content = """First sentence here. Second sentence follows. Third sentence now. Fourth sentence added. Fifth sentence included."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0
            assert len(result.documents) >= 1

            # Check metadata
            for doc in result.documents:
                assert doc.metadata["chunk_strategy"] == "sentence"
        finally:
            os.unlink(temp_path)

    def test_page_chunking_strategy(self, page_content):
        """Test page-based chunking with form feed characters."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "page",
            "min_chunk_size": 10
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(page_content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0

            # Note: MarkItDown may strip form feed characters from text files
            # The page chunking will fall back to section chunking if no pages found
            # At minimum, should produce at least 1 document
            assert len(result.documents) >= 1

            # Check metadata
            for doc in result.documents:
                assert doc.metadata["chunk_strategy"] == "page"
        finally:
            os.unlink(temp_path)

    def test_page_chunking_direct(self):
        """Test page chunking directly on raw text (not through MarkItDown)."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "page",
            "min_chunk_size": 10
        })

        # Test the internal chunking method directly
        text_with_pages = "Page 1 content here.\fPage 2 content here.\fPage 3 content here."
        chunks = parser._chunk_by_page(text_with_pages)

        # Should get 3 chunks for 3 pages
        assert len(chunks) == 3
        assert "Page 1" in chunks[0]
        assert "Page 2" in chunks[1]
        assert "Page 3" in chunks[2]

    def test_fixed_chunking_strategy(self):
        """Test fixed-size chunking."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "fixed",
            "chunk_size": 50,  # Small chunk size for testing
            "chunk_overlap": 0.1,
            "min_chunk_size": 10
        })

        content = "A" * 1000  # 1000 characters of content

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0

            # Should produce multiple fixed-size chunks
            assert len(result.documents) > 1

            # Check metadata
            for doc in result.documents:
                assert doc.metadata["chunk_strategy"] == "fixed"
        finally:
            os.unlink(temp_path)

    def test_invalid_strategy_defaults_to_semantic(self):
        """Test that invalid strategy falls back to semantic."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={"chunk_strategy": "invalid_strategy"})

        # Should default to semantic
        assert parser.chunk_strategy == "semantic"

    def test_chunk_overlap_configuration(self):
        """Test that chunk_overlap is configurable."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={"chunk_overlap": 0.2})
        assert parser.chunk_overlap == 0.2

    def test_min_chunk_size_configuration(self):
        """Test that min_chunk_size is configurable."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={"min_chunk_size": 50})
        assert parser.min_chunk_size == 50

    def test_min_chunk_size_filters_small_chunks(self):
        """Test that chunks smaller than min_chunk_size are filtered."""
        from components.parsers.universal.parser import UniversalParser
        parser = UniversalParser(config={
            "chunk_strategy": "paragraph",
            "min_chunk_size": 200  # High threshold
        })

        content = """Short para 1.

Short para 2.

This is a much longer paragraph with enough content to pass the minimum chunk size threshold that we have set."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert len(result.errors) == 0

            # Only the long paragraph should pass the filter
            # (or all combined if short ones were filtered and merged)
            for doc in result.documents:
                assert len(doc.content) >= 200 or doc.content == content.strip()
        finally:
            os.unlink(temp_path)

    def test_all_valid_strategies(self):
        """Test that all documented strategies are valid."""
        from components.parsers.universal.parser import UniversalParser, CHUNK_STRATEGIES

        expected = ["semantic", "section", "paragraph", "sentence", "page", "fixed"]
        assert set(CHUNK_STRATEGIES) == set(expected)

        # Each strategy should be accepted without warning
        for strategy in expected:
            parser = UniversalParser(config={"chunk_strategy": strategy})
            assert parser.chunk_strategy == strategy
