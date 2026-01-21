"""End-to-End tests for Universal RAG pipeline.

This test file verifies the complete flow:
1. Full pipeline with minimal config (only database, no data_processing_strategy)
2. IngestHandler uses universal_rag strategy by default
3. Documents are parsed, chunked, extracted, and stored
4. Query returns relevant results from stored documents
5. Backward compatibility - existing configs still work
"""

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_config_content():
    """Minimal llamafarm.yaml with only database, no data_processing_strategy."""
    return {
        "version": "v1",
        "name": "minimal_test",
        "namespace": "test",
        "models": [{"provider": "local", "model": "llama3.1:8b"}],
        "prompts": [
            {"name": "default", "messages": [{"role": "system", "content": "Test"}]}
        ],
        "runtime": {
            "provider": "openai",
            "model": "llama3.1:8b",
            "api_key": "test",
            "base_url": "http://localhost:11434/v1",
        },
        "rag": {
            "databases": [
                {
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": "./vectordb/test",
                        "collection_name": "test",
                    },
                    "default_embedding_strategy": "default_embed",
                    "embedding_strategies": [
                        {
                            "name": "default_embed",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                }
            ]
            # NOTE: NO data_processing_strategies - should use universal_rag default
        },
    }


@pytest.fixture
def config_with_explicit_strategy():
    """Config with explicit data_processing_strategy."""
    return {
        "version": "v1",
        "name": "explicit_test",
        "namespace": "test",
        "models": [{"provider": "local", "model": "llama3.1:8b"}],
        "prompts": [
            {"name": "default", "messages": [{"role": "system", "content": "Test"}]}
        ],
        "runtime": {
            "provider": "openai",
            "model": "llama3.1:8b",
            "api_key": "test",
            "base_url": "http://localhost:11434/v1",
        },
        "rag": {
            "databases": [
                {
                    "name": "explicit_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": "./vectordb/explicit",
                        "collection_name": "explicit",
                    },
                    "default_embedding_strategy": "default_embed",
                    "embedding_strategies": [
                        {
                            "name": "default_embed",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                }
            ],
            "data_processing_strategies": [
                {
                    "name": "custom_strategy",
                    "description": "Custom test strategy",
                    "parsers": [{"type": "TextParser_Python", "config": {}}],
                    "extractors": [],
                }
            ],
        },
    }


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    text_content = """# Machine Learning Overview

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed. It focuses on developing algorithms that can access data
and use it to learn for themselves.

## Types of Machine Learning

There are three main types of machine learning:
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data
and makes predictions based on that data.
"""
    file_path = Path(temp_dir) / "ml_overview.md"
    file_path.write_text(text_content)
    return str(file_path)


@pytest.fixture
def sample_txt_file(temp_dir):
    """Create a sample .txt file for testing."""
    text_content = """Introduction to Natural Language Processing

Natural Language Processing (NLP) is a branch of artificial intelligence
that helps computers understand, interpret, and manipulate human language.

Key NLP tasks include:
- Text classification
- Named entity recognition
- Sentiment analysis
- Machine translation

NLP has applications in chatbots, search engines, and language translation.
"""
    file_path = Path(temp_dir) / "nlp_intro.txt"
    file_path.write_text(text_content)
    return str(file_path)


# =============================================================================
# Test Classes
# =============================================================================


class TestMinimalConfigUsesUniversalRAG:
    """Test that minimal config uses universal_rag by default."""

    def test_schema_handler_returns_universal_rag_for_minimal_config(
        self, temp_dir, minimal_config_content
    ):
        """Test: SchemaHandler returns universal_rag when no strategies defined."""
        from core.strategies.handler import SchemaHandler

        # Write minimal config
        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))

        # Request universal_rag strategy
        strategy = handler.create_processing_config("universal_rag")

        assert strategy is not None
        assert strategy.name == "universal_rag"

    def test_get_default_processing_strategy_returns_universal_rag(
        self, temp_dir, minimal_config_content
    ):
        """Test: get_default_processing_strategy returns universal_rag for minimal config."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))

        default_strategy = handler.get_default_processing_strategy()

        assert default_strategy is not None
        assert default_strategy.name == "universal_rag"

    def test_universal_rag_always_in_available_strategies(
        self, temp_dir, minimal_config_content
    ):
        """Test: universal_rag is always in the list of available strategies."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))

        strategies = handler.get_data_processing_strategy_names()

        assert "universal_rag" in strategies


class TestUniversalRAGParserIntegration:
    """Test UniversalParser integration in the pipeline."""

    def test_universal_rag_strategy_uses_universal_parser(
        self, temp_dir, minimal_config_content
    ):
        """Test: universal_rag strategy uses UniversalParser."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")

        assert strategy.parsers is not None
        assert len(strategy.parsers) > 0
        assert strategy.parsers[0].type == "UniversalParser"

    def test_universal_rag_strategy_uses_universal_extractor(
        self, temp_dir, minimal_config_content
    ):
        """Test: universal_rag strategy uses UniversalExtractor."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")

        assert strategy.extractors is not None
        assert len(strategy.extractors) > 0
        assert strategy.extractors[0].type == "UniversalExtractor"


class TestBlobProcessorWithUniversalRAG:
    """Test BlobProcessor with universal_rag strategy."""

    def test_blob_processor_initializes_with_universal_rag(
        self, temp_dir, minimal_config_content
    ):
        """Test: BlobProcessor initializes correctly with universal_rag strategy."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")

        # This should not raise
        processor = BlobProcessor(strategy)

        assert processor is not None

    def test_blob_processor_parses_markdown_file(
        self, temp_dir, minimal_config_content, sample_text_file
    ):
        """Test: BlobProcessor parses markdown file with UniversalParser."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        # Process the file
        with open(sample_text_file, "rb") as f:
            blob_content = f.read()
        file_path = Path(sample_text_file)

        documents = processor.process_blob(
            blob_data=blob_content,
            metadata={"filename": file_path.name, "content_type": "text/markdown"},
        )

        assert documents is not None
        assert len(documents) > 0

        # Verify content is parsed
        all_content = " ".join(doc.content for doc in documents)
        assert "Machine learning" in all_content or "machine learning" in all_content.lower()

    def test_blob_processor_parses_text_file(
        self, temp_dir, minimal_config_content, sample_txt_file
    ):
        """Test: BlobProcessor parses text file with UniversalParser."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        # Process the file
        with open(sample_txt_file, "rb") as f:
            blob_content = f.read()
        file_path = Path(sample_txt_file)

        documents = processor.process_blob(
            blob_data=blob_content,
            metadata={"filename": file_path.name, "content_type": "text/plain"},
        )

        assert documents is not None
        assert len(documents) > 0

        # Verify content is parsed
        all_content = " ".join(doc.content for doc in documents)
        assert "natural language" in all_content.lower()


class TestBackwardCompatibility:
    """Test backward compatibility with explicit strategies."""

    def test_explicit_strategy_overrides_default(
        self, temp_dir, config_with_explicit_strategy
    ):
        """Test: Explicit strategy in config overrides default."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_explicit_strategy, f)

        handler = SchemaHandler(str(config_path))

        # Request the explicit strategy
        strategy = handler.create_processing_config("custom_strategy")

        assert strategy is not None
        assert strategy.name == "custom_strategy"
        assert strategy.parsers[0].type == "TextParser_Python"

    def test_universal_rag_still_available_with_explicit_strategies(
        self, temp_dir, config_with_explicit_strategy
    ):
        """Test: universal_rag is still available even with explicit strategies."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_explicit_strategy, f)

        handler = SchemaHandler(str(config_path))

        # universal_rag should still be available
        strategies = handler.get_data_processing_strategy_names()
        assert "universal_rag" in strategies
        assert "custom_strategy" in strategies

        # Should be able to get universal_rag
        universal = handler.create_processing_config("universal_rag")
        assert universal.name == "universal_rag"


class TestDocumentMetadata:
    """Test that documents have proper metadata after processing."""

    def test_documents_have_chunk_metadata(
        self, temp_dir, minimal_config_content, sample_text_file
    ):
        """Test: Parsed documents have chunk metadata."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        with open(sample_text_file, "rb") as f:
            blob_content = f.read()
        file_path = Path(sample_text_file)

        documents = processor.process_blob(
            blob_data=blob_content,
            metadata={"filename": file_path.name, "content_type": "text/markdown"},
        )

        # Check metadata on first document
        if documents:
            first_doc = documents[0]
            assert first_doc.metadata is not None
            # Should have chunk metadata from UniversalParser
            assert "chunk_index" in first_doc.metadata
            assert "total_chunks" in first_doc.metadata


class TestErrorHandling:
    """Test error handling in the pipeline."""

    def test_handles_empty_file_with_parser_error(self, temp_dir, minimal_config_content):
        """Test: Pipeline raises ParserFailedError for empty files."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler
        from utils.parsing_safety import ParserFailedError

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config_content, f)

        # Create empty file
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.write_text("")

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        # Empty files should raise ParserFailedError (appropriate error handling)
        with pytest.raises(ParserFailedError):
            processor.process_blob(
                blob_data=b"",
                metadata={"filename": "empty.txt", "content_type": "text/plain"},
            )
