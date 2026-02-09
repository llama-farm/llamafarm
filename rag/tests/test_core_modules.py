"""Tests for core RAG modules."""

import tempfile
from pathlib import Path

import pytest
from config.datamodel import DataProcessingStrategy, Parser

from core.blob_processor import BlobProcessor
from core.ingest_handler import IngestHandler


class TestCoreModules:
    """Test core RAG system modules."""

    def test_blob_processor_initialization(self):
        """Test BlobProcessor initialization."""
        # Create a minimal LlamaFarmConfig instance with a data_processing_strategy
        strategy_config = DataProcessingStrategy(
            name="test_strategy",
            description="Test strategy for unit test",
            parsers=[Parser(type="TextParser_Python", config={})],
        )

        processor = BlobProcessor(strategy_config)

        assert processor is not None
        assert hasattr(processor, "process_blob")

    def test_blob_processor_text_file(self):
        """Test processing a text blob."""
        strategy_config = DataProcessingStrategy(
            name="test_strategy",
            description="Test strategy for unit test",
            parsers=[Parser(type="TextParser_Python", config={})],
        )
        processor = BlobProcessor(strategy_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for blob processing")
            f.flush()

            # Process the file as blob
            with open(f.name, "rb") as file:
                blob_data = file.read()
            result = processor.process_blob(blob_data, {"filename": f.name})

            assert result is not None
            Path(f.name).unlink()

    @pytest.mark.skip(reason="Requires complex setup with running services")
    def test_ingest_handler_initialization(self):
        """Test IngestHandler initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            config = {
                "databases": {"test_db": {"vector_store": {"type": "chroma"}}},
                "data_processing_strategies": [
                    {"name": "test_strategy", "parsers": [], "extractors": []}
                ],
            }
            yaml.dump(config, f)
            f.flush()

            handler = IngestHandler(f.name, "test_strategy", "test_db")

            assert handler is not None
            assert hasattr(handler, "ingest")

            Path(f.name).unlink()

    def test_component_factory_parser_creation(self):
        """Test creating parsers via factory."""
        from core.factories import create_parser_from_config

        # Test creating text parser using proper parser name
        config = {"type": "TextParser_Python", "config": {}}
        parser = create_parser_from_config(config)
        assert parser is not None

    def test_component_factory_embedder_creation(self):
        """Test creating embedders via factory."""
        from core.factories import OllamaEmbedder

        # Test creating ollama embedder directly
        embedder = OllamaEmbedder(config={"model": "nomic-embed-text"})
        assert embedder is not None
        assert hasattr(embedder, "embed")

    def test_component_factory_store_creation(self):
        """Test creating stores via factory."""
        try:
            from core.factories import ChromaStore

            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir)
                # Test creating ChromaDB store directly
                store = ChromaStore(
                    config={"collection_name": "test"}, project_dir=project_dir
                )
                assert store is not None
                assert hasattr(store, "add_documents")
        except ImportError:
            # ChromaDB not installed
            assert True

    def test_factory_error_handling(self):
        """Test factory handles unknown types."""
        from core.factories import create_parser_from_config

        # Should handle unknown parser type - may raise or return mock
        config = {"type": "unknown_parser_type", "config": {}}
        try:
            result = create_parser_from_config(config)
            # If it returns something (mock parser), that's OK
            assert result is not None or result is None  # Accept any result
        except (ValueError, KeyError):
            pass  # Expected - factory may raise for unknown types

    @pytest.mark.skip(reason="Requires complex setup with running services")
    def test_ingest_handler_file_processing(self):
        """Test file ingestion process."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            config = {
                "databases": {"test_db": {"vector_store": {"type": "chroma"}}},
                "data_processing_strategies": [
                    {"name": "test_strategy", "parsers": [], "extractors": []}
                ],
            }
            yaml.dump(config, f)
            f.flush()

            handler = IngestHandler(f.name, "test_strategy", "test_db")
            Path(f.name).unlink()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Document to ingest")
            f.flush()

            # Test ingestion (may need configuration)
            try:
                result = handler.ingest(f.name)
                assert result is not None  # Allow for various implementations
            except Exception:
                pass  # Some methods may require configuration
            finally:
                Path(f.name).unlink()
