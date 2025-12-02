"""
Tests for custom RAG query support in chat/completions endpoint.

Tests the rag_queries parameter that allows overriding
the default behavior of using the user message for RAG retrieval.
"""

from unittest.mock import MagicMock, patch

import pytest

from api.routers.projects.projects import ChatRequest
from services.project_chat_service import (
    ProjectChatService,
    RAGParameters,
)


class TestChatRequestModel:
    """Tests for the ChatRequest model with custom RAG query field."""

    def test_chat_request_with_single_rag_query(self):
        """Test ChatRequest accepts a single custom RAG query as array."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Summarize the findings"}],
            rag_queries=["clinical trial results primary endpoints efficacy"],
        )
        assert request.rag_queries == [
            "clinical trial results primary endpoints efficacy"
        ]

    def test_chat_request_with_multiple_rag_queries(self):
        """Test ChatRequest accepts multiple custom RAG queries."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Compare the approaches"}],
            rag_queries=[
                "machine learning methodology",
                "traditional statistical analysis",
            ],
        )
        assert request.rag_queries == [
            "machine learning methodology",
            "traditional statistical analysis",
        ]

    def test_chat_request_without_custom_queries(self):
        """Test ChatRequest works normally without custom queries."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            rag_enabled=True,
        )
        assert request.rag_queries is None


class TestRAGParameters:
    """Tests for RAGParameters with custom query field."""

    def test_rag_parameters_with_single_query(self):
        """Test RAGParameters includes rag_queries field with single item."""
        params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_queries=["my custom query"],
        )
        assert params.rag_queries == ["my custom query"]

    def test_rag_parameters_with_multiple_queries(self):
        """Test RAGParameters includes rag_queries field with multiple items."""
        params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_queries=["query1", "query2"],
        )
        assert params.rag_queries == ["query1", "query2"]

    def test_rag_parameters_defaults(self):
        """Test RAGParameters has None default for rag_queries."""
        params = RAGParameters(rag_enabled=True)
        assert params.rag_queries is None


class TestResolveRAGParameters:
    """Tests for _resolve_rag_parameters with custom query passthrough."""

    @pytest.fixture
    def service(self):
        return ProjectChatService()

    @pytest.fixture
    def mock_config(self):
        """Create a mock project config with RAG enabled."""
        config = MagicMock()
        config.rag = MagicMock()
        config.rag.databases = [MagicMock(name="test_db")]
        config.rag.default_database = "test_db"
        config.rag.databases[0].name = "test_db"
        # Must have at least one retrieval strategy or _resolve_rag_parameters
        # returns early with rag_enabled=False
        strategy = MagicMock()
        strategy.name = "default_strategy"
        strategy.config = None
        config.rag.databases[0].default_retrieval_strategy = "default_strategy"
        config.rag.databases[0].retrieval_strategies = [strategy]
        return config

    def test_resolve_passes_through_rag_queries(self, service, mock_config):
        """Test that rag_queries is passed through to RAGParameters."""
        result = service._resolve_rag_parameters(
            mock_config,
            rag_enabled=True,
            rag_queries=["custom query 1", "custom query 2"],
        )
        assert result.rag_queries == ["custom query 1", "custom query 2"]

    def test_resolve_single_query_in_array(self, service, mock_config):
        """Test that single query in array is passed through."""
        result = service._resolve_rag_parameters(
            mock_config,
            rag_enabled=True,
            rag_queries=["single custom query"],
        )
        assert result.rag_queries == ["single custom query"]


class TestPerformRAGSearch:
    """Tests for _perform_rag_search method (unified RAG search with custom query support)."""

    @pytest.fixture
    def service(self):
        return ProjectChatService()

    @pytest.fixture
    def mock_result(self):
        """Create a mock RAG result."""
        result = MagicMock()
        result.content = "Test content"
        result.metadata = {"source": "test.pdf", "score": 0.9}
        result.score = 0.9
        return result

    @pytest.mark.asyncio
    async def test_uses_single_custom_query_when_provided(self, service, mock_result):
        """Test that single custom query is used instead of user message."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            rag_queries=["custom search query"],
        )

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            mock_search.return_value = [mock_result]

            results = await service._perform_rag_search(
                project_dir="/test/dir",
                message="user message ignored",
                rag_params=rag_params,
            )

            # Verify custom query was used
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs["query"] == "custom search query"
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_uses_user_message_when_no_custom_queries(self, service, mock_result):
        """Test that user message is used when no custom queries."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
        )

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            mock_search.return_value = [mock_result]

            await service._perform_rag_search(
                project_dir="/test/dir",
                message="user message used",
                rag_params=rag_params,
            )

            # Verify user message was used
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs["query"] == "user message used"

    @pytest.mark.asyncio
    async def test_multiple_queries_executed_and_merged(self, service):
        """Test that multiple custom queries are executed and results merged."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            rag_queries=["query1", "query2"],
        )

        # Create different results for each query
        result1 = MagicMock()
        result1.content = "Result from query 1"
        result1.metadata = {"source": "doc1.pdf"}
        result1.score = 0.9

        result2 = MagicMock()
        result2.content = "Result from query 2"
        result2.metadata = {"source": "doc2.pdf"}
        result2.score = 0.8

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            # Return different results for each call
            mock_search.side_effect = [[result1], [result2]]

            results = await service._perform_rag_search(
                project_dir="/test/dir",
                message="ignored",
                rag_params=rag_params,
            )

            # Verify both queries were executed
            assert mock_search.call_count == 2

            # Verify results were merged
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_multiple_queries_deduplicates_results(self, service):
        """Test that duplicate results from multiple queries are deduplicated."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            rag_queries=["query1", "query2"],
        )

        # Create the same result for both queries (same content)
        result = MagicMock()
        result.content = "Same content from both queries"
        result.metadata = {"source": "doc.pdf"}
        result.score = 0.9

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            # Return same result for both queries
            mock_search.return_value = [result]

            results = await service._perform_rag_search(
                project_dir="/test/dir",
                message="ignored",
                rag_params=rag_params,
            )

            # Verify deduplication - should only have 1 result
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_queries_sorted_by_score(self, service):
        """Test that merged results are sorted by score descending."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=10,
            rag_queries=["query1", "query2"],
        )

        # Create results with different scores
        result1 = MagicMock()
        result1.content = "Low score result"
        result1.metadata = {}
        result1.score = 0.5

        result2 = MagicMock()
        result2.content = "High score result"
        result2.metadata = {}
        result2.score = 0.95

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            mock_search.side_effect = [[result1], [result2]]

            results = await service._perform_rag_search(
                project_dir="/test/dir",
                message="ignored",
                rag_params=rag_params,
            )

            # Verify sorted by score descending
            assert len(results) == 2
            assert results[0].score == 0.95
            assert results[1].score == 0.5

    @pytest.mark.asyncio
    async def test_multiple_queries_respects_top_k_limit(self, service):
        """Test that merged results respect the top_k limit."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=2,  # Only want top 2
            rag_queries=["query1", "query2"],
        )

        # Create multiple results
        results_list = []
        for i in range(5):
            r = MagicMock()
            r.content = f"Result {i} - unique content {'x' * i}"
            r.metadata = {}
            r.score = 0.9 - (i * 0.1)
            results_list.append(r)

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            # First query returns 3 results, second returns 2
            mock_search.side_effect = [results_list[:3], results_list[3:]]

            results = await service._perform_rag_search(
                project_dir="/test/dir",
                message="ignored",
                rag_params=rag_params,
            )

            # Verify limited to top_k
            assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_queries_fall_back_to_user_message(self, service, mock_result):
        """Test that empty strings in rag_queries fall back to user message."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            rag_queries=["", "  "],  # All empty
        )

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            mock_search.return_value = [mock_result]

            await service._perform_rag_search(
                project_dir="/test/dir",
                message="fallback to this message",
                rag_params=rag_params,
            )

            # Verify user message was used as fallback
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs["query"] == "fallback to this message"

    @pytest.mark.asyncio
    async def test_mixed_empty_and_valid_queries(self, service, mock_result):
        """Test that empty strings are filtered but valid queries still execute."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            rag_queries=["valid query", "", "  ", "another valid"],
        )

        with patch.object(service, "_execute_single_rag_query") as mock_search:
            mock_search.return_value = [mock_result]

            await service._perform_rag_search(
                project_dir="/test/dir",
                message="ignored",
                rag_params=rag_params,
            )

            # Verify only valid queries were executed (2, not 4)
            assert mock_search.call_count == 2
