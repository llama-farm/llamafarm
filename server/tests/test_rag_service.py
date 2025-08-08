#!/usr/bin/env python3
"""
Comprehensive test suite for the RAGService.
"""

import pytest
from unittest.mock import Mock, patch, call
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the problematic imports at the module level
class MockPipeline:
    def __init__(self, name):
        self.name = name
        self.components = []

    def add_component(self, component):
        self.components.append(component)

    def run(self, source=None):
        return f"result_for_{source}"

class MockCustomerSupportCSVParser:
    def __init__(self, **kwargs):
        self.config = kwargs

class MockOllamaEmbedder:
    def __init__(self, **kwargs):
        self.config = kwargs

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

class MockChromaStore:
    def __init__(self, **kwargs):
        self.config = kwargs

    def search(self, **kwargs):
        return [Mock(score=0.9, content="result1"), Mock(score=0.8, content="result2")]

# Mock the imports before importing RAGService
sys.modules['core.base'] = Mock()
sys.modules['core.base'].Pipeline = MockPipeline
sys.modules['parsers.csv_parser'] = Mock()
sys.modules['parsers.csv_parser'].CustomerSupportCSVParser = MockCustomerSupportCSVParser
sys.modules['embedders.ollama_embedder'] = Mock()
sys.modules['embedders.ollama_embedder'].OllamaEmbedder = MockOllamaEmbedder
sys.modules['stores.chroma_store'] = Mock()
sys.modules['stores.chroma_store'].ChromaStore = MockChromaStore

from services.rag_service import RAGService


class TestRAGService:
    """Test class for RAGService functionality."""

    @pytest.fixture
    def sample_rag_config(self):
        """Sample RAG configuration for testing."""
        return {
            "defaults": {
                "parser": "csv_parser",
                "embedder": "default_embedder",
                "vector_store": "default_store"
            },
            "parsers": {
                "csv_parser": {
                    "type": "CustomerSupportCSVParser",
                    "config": {"delimiter": ",", "encoding": "utf-8"}
                }
            },
            "embedders": {
                "default_embedder": {
                    "type": "OllamaEmbedder",
                    "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}
                }
            },
            "vector_stores": {
                "default_store": {
                    "type": "ChromaStore",
                    "config": {"collection_name": "test_collection", "persist_directory": "/tmp/chroma"}
                }
            }
        }

    @pytest.fixture
    def sample_project_config(self, sample_rag_config):
        """Sample project configuration for testing."""
        return {
            "name": "Test Project",
            "rag": sample_rag_config
        }

    @pytest.fixture
    def sample_dataset_config(self):
        """Sample dataset configuration for testing."""
        return {
            "name": "Test Dataset",
            "files": ["/path/to/file1.csv", "/path/to/file2.csv"],
            "parser": "csv_parser",
            "embedder": "default_embedder",
            "vector_store": "default_store"
        }

    @pytest.fixture
    def rag_service(self):
        """Create RAGService instance for testing."""
        return RAGService()

    def test_get_parser_from_config_success(self, sample_rag_config):
        """Test successful parser instantiation from config."""
        result = RAGService._get_parser_from_config(sample_rag_config, "csv_parser")

        assert isinstance(result, MockCustomerSupportCSVParser)
        assert result.config == {"delimiter": ",", "encoding": "utf-8"}

    def test_get_parser_from_config_missing_parser(self, sample_rag_config):
        """Test error when parser not found in config."""
        with pytest.raises(ValueError, match="Parser 'nonexistent' not found in config"):
            RAGService._get_parser_from_config(sample_rag_config, "nonexistent")

    def test_get_parser_from_config_unsupported_type(self, sample_rag_config):
        """Test error for unsupported parser type."""
        sample_rag_config["parsers"]["bad_parser"] = {
            "type": "UnsupportedParser",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unsupported parser type: UnsupportedParser"):
            RAGService._get_parser_from_config(sample_rag_config, "bad_parser")

    def test_get_embedder_from_config_success(self, sample_rag_config):
        """Test successful embedder instantiation from config."""
        result = RAGService._get_embedder_from_config(sample_rag_config, "default_embedder")

        assert isinstance(result, MockOllamaEmbedder)
        assert result.config == {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}

    def test_get_embedder_from_config_missing_embedder(self, sample_rag_config):
        """Test error when embedder not found in config."""
        with pytest.raises(ValueError, match="Embedder 'nonexistent' not found in config"):
            RAGService._get_embedder_from_config(sample_rag_config, "nonexistent")

    def test_get_embedder_from_config_unsupported_type(self, sample_rag_config):
        """Test error for unsupported embedder type."""
        sample_rag_config["embedders"]["bad_embedder"] = {
            "type": "UnsupportedEmbedder",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unsupported embedder type: UnsupportedEmbedder"):
            RAGService._get_embedder_from_config(sample_rag_config, "bad_embedder")

    def test_get_vector_store_from_config_success(self, sample_rag_config):
        """Test successful vector store instantiation from config."""
        result = RAGService._get_vector_store_from_config(sample_rag_config, "default_store")

        assert isinstance(result, MockChromaStore)
        assert result.config == {"collection_name": "test_collection", "persist_directory": "/tmp/chroma"}

    def test_get_vector_store_from_config_missing_store(self, sample_rag_config):
        """Test error when vector store not found in config."""
        with pytest.raises(ValueError, match="Vector store 'nonexistent' not found in config"):
            RAGService._get_vector_store_from_config(sample_rag_config, "nonexistent")

    def test_get_vector_store_from_config_unsupported_type(self, sample_rag_config):
        """Test error for unsupported vector store type."""
        sample_rag_config["vector_stores"]["bad_store"] = {
            "type": "UnsupportedStore",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unsupported vector store type: UnsupportedStore"):
            RAGService._get_vector_store_from_config(sample_rag_config, "bad_store")

    def test_ingest_dataset_success(self, sample_project_config, sample_dataset_config):
        """Test successful dataset ingestion."""
        result = RAGService.ingest_dataset(sample_project_config, sample_dataset_config)

        # Should return results for each file
        assert len(result) == 2
        assert result[0] == "result_for_/path/to/file1.csv"
        assert result[1] == "result_for_/path/to/file2.csv"

    def test_ingest_dataset_uses_defaults(self, sample_project_config):
        """Test that ingest_dataset uses default values when not specified in dataset config."""
        dataset_config_no_components = {
            "name": "Test Dataset",
            "files": ["/path/to/file.csv"]
        }

        # This should work without error, using defaults from rag config
        result = RAGService.ingest_dataset(sample_project_config, dataset_config_no_components)
        assert len(result) == 1
        assert result[0] == "result_for_/path/to/file.csv"

    def test_retrieve_success(self, rag_service, sample_project_config, sample_dataset_config):
        """Test successful retrieval operation."""
        result = rag_service.retrieve(
            sample_project_config,
            sample_dataset_config,
            "test query",
            top_k=5,
            min_score=0.7
        )

        # Should return mock results
        assert len(result) == 2
        assert result[0].score == 0.9
        assert result[1].score == 0.8

    def test_retrieve_with_optional_parameters(self, rag_service, sample_project_config, sample_dataset_config):
        """Test retrieval with optional parameters."""
        result = rag_service.retrieve(
            sample_project_config,
            sample_dataset_config,
            "test query",
            top_k=10,
            min_score=0.5,
            metadata_filter={"category": "support"},
            return_raw_documents=True
        )

        # Should still return results (mocked)
        assert len(result) == 2

    def test_retrieve_uses_defaults_from_config(self, rag_service, sample_project_config):
        """Test that retrieve uses default values from rag config when not specified in dataset config."""
        dataset_config_no_components = {
            "name": "Test Dataset"
        }

        result = rag_service.retrieve(sample_project_config, dataset_config_no_components, "test query")

        # Should return mock results using defaults
        assert len(result) == 2

    def test_retrieve_minimal_parameters(self, rag_service, sample_project_config, sample_dataset_config):
        """Test retrieval with minimal parameters (only required ones)."""
        result = rag_service.retrieve(sample_project_config, sample_dataset_config, "test query")

        # Should return mock results with default parameters
        assert len(result) == 2

    def test_ingest_dataset_missing_rag_config(self, sample_dataset_config):
        """Test ingest_dataset with missing rag config section."""
        project_config_no_rag = {"name": "Test Project"}

        # This should raise an error since the components won't be found in empty rag config
        with pytest.raises(ValueError):
            RAGService.ingest_dataset(project_config_no_rag, sample_dataset_config)

    def test_retrieve_missing_rag_config(self, rag_service, sample_dataset_config):
        """Test retrieve with missing rag config section."""
        project_config_no_rag = {"name": "Test Project"}

        # This should raise an error since the components won't be found in empty rag config
        with pytest.raises(ValueError):
            rag_service.retrieve(project_config_no_rag, sample_dataset_config, "test")

    def test_component_config_with_empty_config_section(self, sample_rag_config):
        """Test component instantiation when config section is empty."""
        # Test parser with empty config
        sample_rag_config["parsers"]["empty_config_parser"] = {
            "type": "CustomerSupportCSVParser"
            # No config section
        }

        result = RAGService._get_parser_from_config(sample_rag_config, "empty_config_parser")
        assert isinstance(result, MockCustomerSupportCSVParser)
        assert result.config == {}

        # Test embedder with empty config
        sample_rag_config["embedders"]["empty_config_embedder"] = {
            "type": "OllamaEmbedder"
        }

        result = RAGService._get_embedder_from_config(sample_rag_config, "empty_config_embedder")
        assert isinstance(result, MockOllamaEmbedder)
        assert result.config == {}

        # Test vector store with empty config
        sample_rag_config["vector_stores"]["empty_config_store"] = {
            "type": "ChromaStore"
        }

        result = RAGService._get_vector_store_from_config(sample_rag_config, "empty_config_store")
        assert isinstance(result, MockChromaStore)
        assert result.config == {}

    def test_fallback_to_hardcoded_defaults(self):
        """Test fallback behavior when no defaults are provided in config."""
        project_config_minimal = {
            "name": "Test Project",
            "rag": {
                "parsers": {
                    "auto": {
                        "type": "CustomerSupportCSVParser",
                        "config": {}
                    }
                },
                "embedders": {
                    "default": {
                        "type": "OllamaEmbedder",
                        "config": {}
                    }
                },
                "vector_stores": {
                    "default": {
                        "type": "ChromaStore",
                        "config": {}
                    }
                }
            }
        }

        dataset_config_minimal = {
            "name": "Test Dataset",
            "files": ["/path/to/file.csv"]
        }

        # Should use hardcoded defaults: parser="auto", embedder="default", vector_store="default"
        result = RAGService.ingest_dataset(project_config_minimal, dataset_config_minimal)
        assert len(result) == 1
        assert result[0] == "result_for_/path/to/file.csv"