"""Tests for CrossEncoderRerankedStrategy."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List

from components.retrievers.cross_encoder_reranked import CrossEncoderRerankedStrategy
from components.retrievers.base import RetrievalResult
from core.base import Document


@pytest.fixture
def mock_project_dir(tmp_path):
    """Create a mock project directory with config."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create minimal llamafarm.yaml
    config_content = """
version: v1
name: test-project
namespace: default

runtime:
  default_model: test
  models:
    - name: reranker
      provider: ollama
      model: bge-reranker-v2-m3
      base_url: http://localhost:11434/v1
    - name: fast
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1

rag:
  databases:
    - name: test_db
      type: ChromaStore
"""
    (project_dir / "llamafarm.yaml").write_text(config_content)
    return project_dir


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Llama fibers are hollow and provide excellent insulation.",
            metadata={"source": "fiber_guide.pdf", "page": 1},
        ),
        Document(
            id="doc2",
            content="Alpaca fibers are softer and finer than llama fibers.",
            metadata={"source": "fiber_guide.pdf", "page": 2},
        ),
        Document(
            id="doc3",
            content="Both llama and alpaca are members of the camelid family.",
            metadata={"source": "animal_taxonomy.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock_store = Mock()
    mock_store.search = Mock()
    return mock_store


@pytest.fixture
def mock_base_strategy(sample_documents):
    """Create a mock base strategy that returns sample documents."""
    mock_strategy = Mock()
    mock_strategy.retrieve = Mock(
        return_value=RetrievalResult(
            documents=sample_documents,
            scores=[0.9, 0.8, 0.7],
            strategy_metadata={"strategy": "BasicSimilarityStrategy"},
        )
    )
    return mock_strategy


class TestCrossEncoderRerankedStrategy:
    """Test suite for CrossEncoderRerankedStrategy."""

    def test_initialization(self, mock_project_dir):
        """Test strategy initialization with config."""
        config = {
            "model_name": "reranker",
            "initial_k": 30,
            "final_k": 10,
            "batch_size": 32,
        }

        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config=config,
            project_dir=mock_project_dir,
        )

        assert strategy.name == "test_reranker"
        assert strategy.model_name == "reranker"
        assert strategy.initial_k == 30
        assert strategy.final_k == 10
        assert strategy.batch_size == 32

    def test_initialization_with_defaults(self, mock_project_dir):
        """Test strategy initialization with default config."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        assert strategy.model_name == "reranker"
        assert strategy.initial_k == 30
        assert strategy.final_k == 10
        assert strategy.batch_size == 32
        assert strategy.normalize_scores == True

    def test_validate_config_valid(self, mock_project_dir):
        """Test config validation with valid config."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config={
                "initial_k": 30,
                "final_k": 10,
                "batch_size": 16,
                "relevance_threshold": 0.5,
            },
            project_dir=mock_project_dir,
        )

        assert strategy.validate_config() == True

    def test_validate_config_invalid(self, mock_project_dir):
        """Test config validation with invalid config."""
        # Test invalid initial_k
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config={"initial_k": 0},
            project_dir=mock_project_dir,
        )
        assert strategy.validate_config() == False

        # Test invalid batch_size
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config={"batch_size": 0},
            project_dir=mock_project_dir,
        )
        assert strategy.validate_config() == False

    def test_supports_vector_store(self, mock_project_dir):
        """Test that strategy supports all vector store types."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        assert strategy.supports_vector_store("ChromaStore") == True
        assert strategy.supports_vector_store("FAISSStore") == True
        assert strategy.supports_vector_store("QdrantStore") == True

    def test_cosine_similarity(self, mock_project_dir):
        """Test cosine similarity calculation."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = strategy._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)

        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = strategy._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0)

        # Test opposite vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = strategy._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0)

    def test_normalize_results(self, mock_project_dir, sample_documents):
        """Test score normalization."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        results = [
            (sample_documents[0], 0.8),
            (sample_documents[1], 0.6),
            (sample_documents[2], 0.4),
        ]

        normalized = strategy._normalize_results(results)

        # Check that scores are normalized to 0-1
        assert normalized[0][1] == pytest.approx(1.0)  # Max score
        assert normalized[1][1] == pytest.approx(0.5)  # Mid score
        assert normalized[2][1] == pytest.approx(0.0)  # Min score

    def test_normalize_results_identical_scores(self, mock_project_dir, sample_documents):
        """Test score normalization when all scores are identical."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        results = [
            (sample_documents[0], 0.5),
            (sample_documents[1], 0.5),
            (sample_documents[2], 0.5),
        ]

        normalized = strategy._normalize_results(results)

        # All should be 0.5 (neutral)
        assert all(score == 0.5 for _, score in normalized)

    @pytest.mark.skip(reason="Integration test - requires server dependencies")
    def test_initialize_reranker(self, mock_project_dir):
        """Test reranker initialization from project config.

        NOTE: This is an integration test that will be covered in end-to-end tests.
        Unit testing this requires complex mocking of server dependencies.
        """
        pass

    def test_initialize_reranker_model_not_found(self, mock_project_dir):
        """Test error when model name not found in config."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config={"model_name": "nonexistent_model"},
            project_dir=mock_project_dir,
        )

        with pytest.raises(ValueError, match="Model configuration not resolved for 'nonexistent_model'"):
            strategy._initialize_reranker()

    def test_retrieve_requires_query_text(
        self, mock_project_dir, mock_vector_store, mock_base_strategy
    ):
        """Test that retrieve requires query_text parameter."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(ValueError, match="query_text is required"):
            strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=mock_vector_store,
                top_k=5,
                query_text="",  # Empty query text should raise error
            )

    def test_get_config_schema(self, mock_project_dir):
        """Test configuration schema retrieval."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            project_dir=mock_project_dir,
        )

        schema = strategy.get_config_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "model_name" in schema["properties"]
        assert "initial_k" in schema["properties"]
        assert "batch_size" in schema["properties"]

    def test_get_performance_info(self, mock_project_dir):
        """Test performance info retrieval."""
        strategy = CrossEncoderRerankedStrategy(
            name="test_reranker",
            config={"model_name": "reranker"},
            project_dir=mock_project_dir,
        )

        perf_info = strategy.get_performance_info()

        assert perf_info["speed"] == "fast"
        assert perf_info["accuracy"] == "very_high"
        assert "simple_questions" in perf_info["best_for"]
        assert "reranker" in perf_info["notes"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
