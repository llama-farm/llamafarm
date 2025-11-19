"""Tests for MultiTurnRAGStrategy."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from components.retrievers.multi_turn.multi_turn import MultiTurnRAGStrategy
from components.retrievers.base import RetrievalResult
from core.base import Document


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.search = Mock(return_value=([], []))
    return store


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(id="doc1", content="Llama fiber is soft and warm.", metadata={}),
        Document(id="doc2", content="Alpaca fiber is hypoallergenic.", metadata={}),
        Document(id="doc3", content="Both are South American camelids.", metadata={}),
    ]


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return [0.1] * 768


class TestMultiTurnRAGStrategy:
    """Tests for MultiTurnRAGStrategy."""

    def test_initialization(self):
        """Test strategy initialization with default config."""
        strategy = MultiTurnRAGStrategy()

        assert strategy.name == "MultiTurnRAGStrategy"
        assert strategy.max_sub_queries == 3
        assert strategy.complexity_threshold == 50
        assert strategy.enable_reranking is False
        assert strategy.max_workers == 3

    def test_initialization_with_config(self):
        """Test strategy initialization with custom config."""
        config = {
            "model_name": "test_model",
            "model_base_url": "http://localhost:8000",
            "model_id": "test/model:latest",
            "max_sub_queries": 5,
            "complexity_threshold": 100,
            "enable_reranking": True,
            "max_workers": 5,
        }

        strategy = MultiTurnRAGStrategy(config=config)

        assert strategy.model_name == "test_model"
        assert strategy.model_base_url == "http://localhost:8000"
        assert strategy.model_id == "test/model:latest"
        assert strategy.max_sub_queries == 5
        assert strategy.complexity_threshold == 100
        assert strategy.enable_reranking is True
        assert strategy.max_workers == 5

    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        strategy = MultiTurnRAGStrategy(
            config={
                "max_sub_queries": 3,
                "sub_query_top_k": 10,
                "final_top_k": 5,
            }
        )

        assert strategy.validate_config() is True

    def test_validate_config_invalid_max_sub_queries(self):
        """Test config validation with invalid max_sub_queries."""
        strategy = MultiTurnRAGStrategy(config={"max_sub_queries": 10})
        assert strategy.validate_config() is False

        strategy = MultiTurnRAGStrategy(config={"max_sub_queries": 0})
        assert strategy.validate_config() is False

    def test_supports_vector_store(self):
        """Test vector store support."""
        strategy = MultiTurnRAGStrategy()
        assert strategy.supports_vector_store("ChromaStore") is True
        assert strategy.supports_vector_store("FAISSStore") is True
        assert strategy.supports_vector_store("AnyStore") is True

    def test_detect_query_complexity_simple(self):
        """Test complexity detection for simple queries."""
        strategy = MultiTurnRAGStrategy(config={"complexity_threshold": 50})

        # Short query
        assert strategy._detect_query_complexity("What is AI?") is False

        # No complexity markers
        assert strategy._detect_query_complexity("Explain machine learning") is False

    def test_detect_query_complexity_complex(self):
        """Test complexity detection for complex queries."""
        strategy = MultiTurnRAGStrategy(config={"complexity_threshold": 50})

        # Long query with "and"
        query = "What are the differences between llama and alpaca fibers?"
        assert strategy._detect_query_complexity(query) is True

        # Multiple questions
        query = "What is AI? How does it work? What are the applications?"
        assert strategy._detect_query_complexity(query) is True

        # Contains "also"
        query = "Explain neural networks and also describe their applications in computer vision"
        assert strategy._detect_query_complexity(query) is True

    def test_decompose_query(self):
        """Test query decomposition."""
        with patch("openai.OpenAI") as mock_openai:
            # Mock LLM response with XML format
            mock_response = Mock()
            mock_response.choices = [
                Mock(message=Mock(content='<question>What is llama fiber?</question>\n<question>What is alpaca fiber?</question>'))
            ]
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            strategy = MultiTurnRAGStrategy(
                config={
                    "model_base_url": "http://localhost:8000",
                    "model_id": "test/model",
                    "max_sub_queries": 3,
                    "min_query_length": 10,
                }
            )

            query = "What are the differences between llama and alpaca fibers?"
            sub_queries = strategy._decompose_query(query)

            assert len(sub_queries) == 2
            assert "What is llama fiber?" in sub_queries
            assert "What is alpaca fiber?" in sub_queries

    def test_decompose_query_fallback_on_error(self):
        """Test query decomposition falls back to original query on error."""
        with patch("openai.OpenAI") as mock_openai:
            # Mock LLM to raise an exception
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_openai.return_value = mock_client

            strategy = MultiTurnRAGStrategy(
                config={
                    "model_base_url": "http://localhost:8000",
                    "model_id": "test/model",
                }
            )

            query = "What are the differences between llama and alpaca fibers?"
            sub_queries = strategy._decompose_query(query)

            # Should return original query as fallback
            assert len(sub_queries) == 1
            assert sub_queries[0] == query

    def test_merge_and_deduplicate(self, sample_documents):
        """Test merging and deduplication of results."""
        strategy = MultiTurnRAGStrategy(config={"final_top_k": 2})

        # Create results from multiple sub-queries
        results = [
            (
                "sub_query_1",
                RetrievalResult(
                    documents=[sample_documents[0], sample_documents[1]],
                    scores=[0.9, 0.8],
                    strategy_metadata={}
                ),
            ),
            (
                "sub_query_2",
                RetrievalResult(
                    documents=[sample_documents[1], sample_documents[2]],  # doc1 is duplicate
                    scores=[0.85, 0.7],
                    strategy_metadata={}
                ),
            ),
        ]

        merged = strategy._merge_and_deduplicate(results, top_k=5)

        # Should have 3 unique documents (doc1, doc2, doc3)
        assert len(merged.documents) <= strategy.final_top_k
        assert len(merged.documents) == len(merged.scores)

        # Check metadata
        assert merged.strategy_metadata["strategy"] == "MultiTurnRAGStrategy"
        assert merged.strategy_metadata["sub_queries_count"] == 2

    def test_retrieve_simple_query(self, mock_vector_store, sample_embedding, sample_documents):
        """Test retrieval for a simple query (no decomposition)."""
        with patch.object(
            MultiTurnRAGStrategy, "_initialize_base_strategy"
        ) as mock_init:
            # Mock base strategy
            mock_base = Mock()
            mock_base.retrieve.return_value = RetrievalResult(
                documents=sample_documents[:2],
                scores=[0.9, 0.8],
                strategy_metadata={}
            )

            strategy = MultiTurnRAGStrategy(
                config={
                    "complexity_threshold": 100,  # High threshold to avoid decomposition
                }
            )
            strategy._base_strategy = mock_base

            result = strategy.retrieve(
                query_embedding=sample_embedding,
                vector_store=mock_vector_store,
                top_k=5,
                query_text="What is AI?",
            )

            # Should use base strategy directly
            assert len(result.documents) == 2
            assert result.strategy_metadata["strategy"] == "MultiTurnRAGStrategy"
            assert result.strategy_metadata["decomposed"] is False

    def test_retrieve_complex_query_with_decomposition(
        self, mock_vector_store, sample_embedding, sample_documents
    ):
        """Test retrieval for a complex query with decomposition."""
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor, \
             patch("openai.OpenAI") as mock_openai:

            # Mock LLM response for decomposition (XML format)
            mock_llm_response = Mock()
            mock_llm_response.choices = [
                Mock(message=Mock(content='<question>What is llama fiber?</question>\n<question>What is alpaca fiber?</question>'))
            ]
            mock_llm_client = Mock()
            mock_llm_client.chat.completions.create.return_value = mock_llm_response
            mock_openai.return_value = mock_llm_client

            # Mock ThreadPoolExecutor
            mock_future = Mock()
            mock_future.result.return_value = (
                "sub_query",
                RetrievalResult(documents=sample_documents[:2], scores=[0.9, 0.8], strategy_metadata={}),
            )
            mock_pool = Mock()
            mock_pool.__enter__ = Mock(return_value=mock_pool)
            mock_pool.__exit__ = Mock(return_value=False)
            mock_pool.submit.return_value = mock_future
            mock_executor.return_value = mock_pool

            # Mock base strategy
            mock_base = Mock()
            mock_base.retrieve.return_value = RetrievalResult(
                documents=sample_documents[:2],
                scores=[0.9, 0.8],
                strategy_metadata={}
            )

            strategy = MultiTurnRAGStrategy(
                config={
                    "model_base_url": "http://localhost:8000",
                    "model_id": "test/model",
                    "complexity_threshold": 30,  # Low threshold to trigger decomposition
                    "max_sub_queries": 2,
                }
            )
            strategy._base_strategy = mock_base

            result = strategy.retrieve(
                query_embedding=sample_embedding,
                vector_store=mock_vector_store,
                top_k=5,
                query_text="What are the differences between llama and alpaca fibers?",
                embedder=Mock(),  # Add embedder for new requirement
            )

            # Should decompose and merge
            assert result.strategy_metadata["decomposed"] is True
            assert "sub_queries" in result.strategy_metadata

    def test_retrieve_requires_query_text(self, mock_vector_store, sample_embedding):
        """Test that retrieve requires query_text."""
        strategy = MultiTurnRAGStrategy()

        with pytest.raises(ValueError, match="query_text is required"):
            strategy.retrieve(
                query_embedding=sample_embedding,
                vector_store=mock_vector_store,
                top_k=5,
                query_text="",  # Empty query text
            )

    def test_get_config_schema(self):
        """Test config schema retrieval."""
        strategy = MultiTurnRAGStrategy()
        schema = strategy.get_config_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "model_name" in schema["properties"]
        assert "max_sub_queries" in schema["properties"]
        assert "enable_reranking" in schema["properties"]

    def test_get_performance_info(self):
        """Test performance info retrieval."""
        strategy = MultiTurnRAGStrategy(config={"model_name": "test_model"})
        info = strategy.get_performance_info()

        assert info["speed"] == "medium-slow"
        assert info["complexity"] == "high"
        assert info["accuracy"] == "very_high"
        assert "complex_queries" in info["best_for"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
