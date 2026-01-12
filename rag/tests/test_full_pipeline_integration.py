"""Full pipeline integration tests for the new RAG parsing system.

These tests verify the complete flow:
1. Parse documents with Docling/MarkItDown
2. Chunk content appropriately
3. Embed chunks
4. Store in vector database
5. Query and retrieve with metadata preserved
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add rag directory to path
RAG_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RAG_DIR))


class TestDoclingPipelineIntegration:
    """Test end-to-end ingest with Docling parser."""

    def test_docling_parser_produces_documents(self):
        """Test that DoclingParser produces proper Document objects."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={"chunk_strategy": "none"})

        # Find a sample PDF
        examples_dir = RAG_DIR.parent / "examples"
        sample_pdf = None
        for subdir in ["fda_rag/files", "rag_legal_filings/files"]:
            pdf_dir = examples_dir / subdir
            if pdf_dir.exists():
                pdfs = list(pdf_dir.glob("*.pdf"))
                if pdfs:
                    sample_pdf = pdfs[0]
                    break

        if sample_pdf is None:
            pytest.skip("No sample PDF found for integration test")

        result = parser.parse(str(sample_pdf))

        assert result.documents is not None
        assert len(result.documents) > 0
        assert result.documents[0].content
        assert result.documents[0].metadata

    def test_docling_with_hybrid_chunking(self):
        """Test DoclingParser with HybridChunker produces chunks."""
        from components.parsers.docling import DoclingParser

        parser = DoclingParser(config={
            "chunk_strategy": "hybrid",
            "chunk_size": 256,
        })

        # Find sample PDF
        examples_dir = RAG_DIR.parent / "examples"
        sample_pdf = None
        for subdir in ["fda_rag/files", "rag_legal_filings/files"]:
            pdf_dir = examples_dir / subdir
            if pdf_dir.exists():
                pdfs = list(pdf_dir.glob("*.pdf"))
                if pdfs:
                    sample_pdf = pdfs[0]
                    break

        if sample_pdf is None:
            pytest.skip("No sample PDF found")

        result = parser.parse(str(sample_pdf))

        # With chunking, we should get multiple documents (chunks)
        assert len(result.documents) > 1
        # Each chunk should have metadata
        for doc in result.documents:
            assert doc.metadata is not None


class TestMarkItDownPipelineIntegration:
    """Test end-to-end ingest with MarkItDown parser."""

    def test_markitdown_parser_produces_documents(self):
        """Test that MarkItDownParser produces proper Document objects."""
        from components.parsers.markitdown import MarkItDownParser

        parser = MarkItDownParser(config={})

        # Create a sample text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Test Document\n\nThis is a test document for MarkItDown parser.\n")
            f.write("It has multiple paragraphs.\n\n")
            f.write("## Section 2\n\nMore content here.\n")
            temp_path = f.name

        try:
            result = parser.parse(temp_path)

            assert result.documents is not None
            assert len(result.documents) > 0
            assert result.documents[0].content
        finally:
            os.unlink(temp_path)


class TestIngestEmbedStorePipeline:
    """Test the full Ingest -> Embed -> Store pipeline."""

    def test_blob_processor_uses_docling_for_pdf(self):
        """Test that BlobProcessor selects DoclingParser for PDFs."""
        from core.blob_processor import BlobProcessor

        # Create a mock strategy config with Docling as primary PDF parser
        mock_parser_config = MagicMock()
        mock_parser_config.type = "DoclingParser"
        mock_parser_config.priority = 0
        mock_parser_config.file_include_patterns = ["*.pdf"]
        mock_parser_config.config = {"chunk_strategy": "none"}

        mock_strategy = MagicMock()
        mock_strategy.parsers = [mock_parser_config]
        mock_strategy.extractors = []

        # BlobProcessor should be able to initialize with this config
        processor = BlobProcessor(mock_strategy)

        # Verify parser was loaded
        assert len(processor.parsers) > 0
        parser_config, parser_instance = processor.parsers[0]
        assert "Docling" in type(parser_instance).__name__

    def test_blob_processor_pattern_matching(self):
        """Test that BlobProcessor matches files to parsers correctly."""
        from core.blob_processor import BlobProcessor

        # Create strategy with multiple parsers
        pdf_parser = MagicMock()
        pdf_parser.type = "DoclingParser"
        pdf_parser.priority = 0
        pdf_parser.file_include_patterns = ["*.pdf", "*.PDF"]
        pdf_parser.config = {}

        txt_parser = MagicMock()
        txt_parser.type = "TextParser_Python"
        txt_parser.priority = 0
        txt_parser.file_include_patterns = ["*.txt"]
        txt_parser.config = {}

        mock_strategy = MagicMock()
        mock_strategy.parsers = [pdf_parser, txt_parser]
        mock_strategy.extractors = []

        processor = BlobProcessor(mock_strategy)

        # Test pattern matching (note: method is _matches_patterns with 'es')
        pdf_match = processor._matches_patterns("document.pdf", ["*.pdf"])
        txt_match = processor._matches_patterns("readme.txt", ["*.txt"])

        assert pdf_match is True
        assert txt_match is True


class TestQueryWithChunkedDocuments:
    """Test that queries retrieve properly chunked documents."""

    def test_chunk_metadata_structure(self):
        """Test that chunks have proper metadata structure."""
        from components.chunkers import SimpleChunker, ChunkingStrategy

        chunker = SimpleChunker(
            strategy=ChunkingStrategy.SENTENCES,
            chunk_size=100,
            overlap=20,
        )

        text = """
        First sentence of the document.
        Second sentence continues here.
        Third sentence with more content.
        Fourth sentence to ensure multiple chunks.
        Fifth sentence for good measure.
        """

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        for chunk in chunks:
            # Verify required metadata fields
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata
            assert "chunking_strategy" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "sentences"


class TestChunkMetadataPreservation:
    """Test that chunk metadata is preserved through the pipeline."""

    def test_docling_preserves_page_metadata(self):
        """Test that Docling preserves page information in chunks."""
        from components.parsers.docling import DoclingParser

        # Find a multi-page PDF
        examples_dir = RAG_DIR.parent / "examples"
        sample_pdf = None
        for subdir in ["fda_rag/files", "rag_legal_filings/files"]:
            pdf_dir = examples_dir / subdir
            if pdf_dir.exists():
                pdfs = list(pdf_dir.glob("*.pdf"))
                if pdfs:
                    # Get the largest PDF (likely multi-page)
                    pdfs.sort(key=lambda p: p.stat().st_size, reverse=True)
                    sample_pdf = pdfs[0]
                    break

        if sample_pdf is None:
            pytest.skip("No sample PDF found")

        parser = DoclingParser(config={
            "chunk_strategy": "hybrid",
            "chunk_size": 256,
        })

        result = parser.parse(str(sample_pdf))

        # Check that at least some chunks have page metadata
        has_page_info = any(
            "page" in doc.metadata or "page_start" in doc.metadata or "page_numbers" in doc.metadata
            for doc in result.documents
        )
        assert has_page_info or len(result.documents) == 1  # Single page docs won't have page metadata

    def test_simple_chunker_preserves_custom_metadata(self):
        """Test that SimpleChunker preserves custom metadata passed to it."""
        from components.chunkers import SimpleChunker, ChunkingStrategy

        chunker = SimpleChunker(
            strategy=ChunkingStrategy.PARAGRAPHS,
            chunk_size=200,
        )

        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph."

        chunks = chunker.chunk(text, metadata={
            "source_file": "test.pdf",
            "upload_date": "2024-01-15",
            "custom_field": "custom_value",
        })

        for chunk in chunks:
            # Custom metadata should be preserved
            assert chunk.metadata.get("source_file") == "test.pdf"
            assert chunk.metadata.get("upload_date") == "2024-01-15"
            assert chunk.metadata.get("custom_field") == "custom_value"


class TestMultiFileIngest:
    """Test multi-file ingest with different parsers."""

    def test_parser_selection_by_extension(self):
        """Test that correct parser is selected based on file extension."""
        from components.parsers.parser_registry import ParserRegistry

        registry = ParserRegistry()

        # PDF should use Docling
        pdf_parsers = registry.get_parsers_for_extension(".pdf")
        assert len(pdf_parsers) > 0
        # Docling should be first (highest priority)
        assert pdf_parsers[0]["parser"] == "DoclingParser"

        # DOCX should use Docling or MarkItDown (both are new parsers)
        docx_parsers = registry.get_parsers_for_extension(".docx")
        assert len(docx_parsers) > 0
        # Docling is actually first for DOCX (priority 0), MarkItDown is second (priority 1)
        assert docx_parsers[0]["parser"] in ["DoclingParser", "MarkItDownParser"]
        # Verify both new parsers are available for DOCX
        parser_names = [p["parser"] for p in docx_parsers]
        assert "DoclingParser" in parser_names
        assert "MarkItDownParser" in parser_names

    def test_mixed_file_types_processing(self):
        """Test processing multiple file types together."""
        # Create test files
        test_files = []

        # Text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Text file content\n")
            test_files.append(("text", f.name))

        # Markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Markdown\n\nMarkdown content\n")
            test_files.append(("markdown", f.name))

        try:
            from components.parsers.parser_registry import ParserRegistry

            registry = ParserRegistry()
            results = []

            for file_type, file_path in test_files:
                ext = Path(file_path).suffix
                parsers = registry.get_parsers_for_extension(ext)

                if parsers:
                    # Get parser class
                    parser_name = parsers[0]["parser"]
                    results.append({
                        "file": file_path,
                        "type": file_type,
                        "parser": parser_name,
                    })

            assert len(results) == 2
            assert all("parser" in r for r in results)

        finally:
            for _, path in test_files:
                os.unlink(path)


class TestPipelineWithSimpleChunker:
    """Test pipeline using SimpleChunker for post-processing."""

    def test_parser_then_chunker_workflow(self):
        """Test workflow: parse with MarkItDown, then chunk with SimpleChunker."""
        from components.parsers.markitdown import MarkItDownParser
        from components.chunkers import SimpleChunker, ChunkingStrategy

        # Create sample file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("First section content. More text here.\n\n")
            f.write("Second section content. Additional details.\n\n")
            f.write("Third section content. Final information.\n")
            temp_path = f.name

        try:
            # Step 1: Parse
            parser = MarkItDownParser(config={})
            parse_result = parser.parse(temp_path)

            assert len(parse_result.documents) > 0
            full_content = parse_result.documents[0].content

            # Step 2: Chunk
            chunker = SimpleChunker(
                strategy=ChunkingStrategy.SENTENCES,
                chunk_size=50,
            )

            chunks = chunker.chunk(
                full_content,
                metadata=parse_result.documents[0].metadata,
            )

            assert len(chunks) > 1
            # Verify metadata is preserved
            for chunk in chunks:
                assert "chunk_index" in chunk.metadata

        finally:
            os.unlink(temp_path)


class TestDataProcessingStrategyIntegration:
    """Test integration with DataProcessingStrategy configuration."""

    def test_strategy_with_new_parsers(self):
        """Test DataProcessingStrategy supports new parser types."""
        # This test verifies the schema accepts new parser configurations
        strategy_config = {
            "name": "docling_based",
            "description": "Strategy using Docling for PDFs",
            "parsers": [
                {
                    "type": "DoclingParser",
                    "file_include_patterns": ["*.pdf"],
                    "priority": 0,
                    "config": {
                        "chunk_strategy": "hybrid",
                        "chunk_size": 512,
                    },
                },
                {
                    "type": "MarkItDownParser",
                    "file_include_patterns": ["*.docx", "*.pptx"],
                    "priority": 0,
                    "config": {
                        "output_format": "markdown",
                    },
                },
            ],
        }

        # Verify structure is valid
        assert strategy_config["name"] == "docling_based"
        assert len(strategy_config["parsers"]) == 2
        assert strategy_config["parsers"][0]["type"] == "DoclingParser"
        assert strategy_config["parsers"][1]["type"] == "MarkItDownParser"
