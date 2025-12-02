"""
Tests for custom RAG query support in chat/completions endpoint.

Tests the rag_query and rag_queries parameters that allow overriding
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
    """Tests for the ChatRequest model with custom RAG query fields."""

    def test_chat_request_with_custom_rag_query(self):
        """Test ChatRequest accepts a single custom RAG query."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Summarize the findings"}],
            rag_query="clinical trial results primary endpoints efficacy",
        )
        assert request.rag_query == "clinical trial results primary endpoints efficacy"
        assert request.rag_queries is None

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
        assert request.rag_query is None

    def test_chat_request_with_both_rag_query_types(self):
        """Test ChatRequest when both rag_query and rag_queries are provided."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            rag_query="single query",
            rag_queries=["multi1", "multi2"],
        )
        # Both fields should be set - precedence is handled in the service
        assert request.rag_query == "single query"
        assert request.rag_queries == ["multi1", "multi2"]

    def test_chat_request_without_custom_queries(self):
        """Test ChatRequest works normally without custom queries."""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            rag_enabled=True,
        )
        assert request.rag_query is None
        assert request.rag_queries is None


class TestRAGParameters:
    """Tests for RAGParameters with custom query fields."""

    def test_rag_parameters_with_custom_query(self):
        """Test RAGParameters includes custom_query field."""
        params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            custom_query="my custom query",
        )
        assert params.custom_query == "my custom query"
        assert params.custom_queries is None

    def test_rag_parameters_with_custom_queries(self):
        """Test RAGParameters includes custom_queries field."""
        params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            custom_queries=["query1", "query2"],
        )
        assert params.custom_queries == ["query1", "query2"]
        assert params.custom_query is None

    def test_rag_parameters_defaults(self):
        """Test RAGParameters has None defaults for custom queries."""
        params = RAGParameters(rag_enabled=True)
        assert params.custom_query is None
        assert params.custom_queries is None


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
        config.rag.databases[0].name = "test_db"
        config.rag.databases[0].retrieval_strategies = [MagicMock(name="default")]
        config.rag.databases[0].retrieval_strategies[0].name = "default"
        config.rag.databases[0].retrieval_strategies[0].config = None
        config.rag.databases[0].default_retrieval_strategy = None
        config.rag.default_database = None
        return config

    def test_resolve_passes_through_custom_query(self, service, mock_config):
        """Test that custom_query is passed through to RAGParameters."""
        result = service._resolve_rag_parameters(
            mock_config,
            rag_enabled=True,
            database="test_db",
            rag_query="my custom query",
        )
        assert result.rag_enabled is True
        assert result.custom_query == "my custom query"
        assert result.custom_queries is None

    def test_resolve_passes_through_custom_queries(self, service, mock_config):
        """Test that custom_queries is passed through to RAGParameters."""
        result = service._resolve_rag_parameters(
            mock_config,
            rag_enabled=True,
            database="test_db",
            rag_queries=["query1", "query2"],
        )
        assert result.rag_enabled is True
        assert result.custom_queries == ["query1", "query2"]
        assert result.custom_query is None


class TestPerformRAGSearchWithCustomQueries:
    """Tests for _perform_rag_search_with_custom_queries."""

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

    def test_uses_custom_query_when_provided(self, service, mock_result):
        """Test that custom_query is used instead of user message."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            custom_query="custom search query",
        )

        with patch.object(service, "_perform_rag_search") as mock_search:
            mock_search.return_value = [mock_result]

            results = service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="user message ignored",
                rag_params=rag_params,
            )

            # Verify custom query was used
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs["message"] == "custom search query"
            assert len(results) == 1

    def test_uses_user_message_when_no_custom_query(self, service, mock_result):
        """Test that user message is used when no custom query."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
        )

        with patch.object(service, "_perform_rag_search") as mock_search:
            mock_search.return_value = [mock_result]

            service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="user message used",
                rag_params=rag_params,
            )

            # Verify user message was used
            mock_search.assert_called_once()
            call_kwargs = mock_search.call_args.kwargs
            assert call_kwargs["message"] == "user message used"

    def test_multiple_queries_executed_and_merged(self, service):
        """Test that multiple custom queries are executed and results merged."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            custom_queries=["query1", "query2"],
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

        with patch.object(service, "_perform_rag_search") as mock_search:
            # Return different results for each call
            mock_search.side_effect = [[result1], [result2]]

            results = service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="ignored",
                rag_params=rag_params,
            )

            # Verify both queries were executed
            assert mock_search.call_count == 2
            calls = mock_search.call_args_list
            assert calls[0].kwargs["message"] == "query1"
            assert calls[1].kwargs["message"] == "query2"

            # Verify results were merged
            assert len(results) == 2

    def test_multiple_queries_deduplicates_results(self, service):
        """Test that duplicate results from multiple queries are deduplicated."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            custom_queries=["query1", "query2"],
        )

        # Create the same result for both queries (same content)
        result = MagicMock()
        result.content = "Same content from both queries"
        result.metadata = {"source": "doc.pdf"}
        result.score = 0.9

        with patch.object(service, "_perform_rag_search") as mock_search:
            # Return same result for both queries
            mock_search.return_value = [result]

            results = service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="ignored",
                rag_params=rag_params,
            )

            # Verify deduplication - should only have 1 result
            assert len(results) == 1

    def test_multiple_queries_sorted_by_score(self, service):
        """Test that merged results are sorted by score descending."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=10,
            custom_queries=["query1", "query2"],
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

        with patch.object(service, "_perform_rag_search") as mock_search:
            mock_search.side_effect = [[result1], [result2]]

            results = service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="ignored",
                rag_params=rag_params,
            )

            # Verify sorted by score descending
            assert len(results) == 2
            assert results[0].score == 0.95
            assert results[1].score == 0.5

    def test_multiple_queries_respects_top_k_limit(self, service):
        """Test that merged results respect the top_k limit."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=2,  # Only want top 2
            custom_queries=["query1", "query2"],
        )

        # Create multiple results
        results_list = []
        for i in range(5):
            r = MagicMock()
            r.content = f"Result {i} - unique content {'x' * i}"
            r.metadata = {}
            r.score = 0.9 - (i * 0.1)
            results_list.append(r)

        with patch.object(service, "_perform_rag_search") as mock_search:
            # First query returns 3 results, second returns 2
            mock_search.side_effect = [results_list[:3], results_list[3:]]

            results = service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="ignored",
                rag_params=rag_params,
            )

            # Verify limited to top_k
            assert len(results) == 2

    def test_custom_queries_takes_precedence_over_custom_query(
        self, service, mock_result
    ):
        """Test that custom_queries takes precedence when both are set."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            custom_query="single query ignored",
            custom_queries=["multi1", "multi2"],
        )

        with patch.object(service, "_perform_rag_search") as mock_search:
            mock_search.return_value = [mock_result]

            service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="user message",
                rag_params=rag_params,
            )

            # Verify custom_queries was used (2 calls), not custom_query
            assert mock_search.call_count == 2

    def test_empty_queries_skipped(self, service, mock_result):
        """Test that empty strings in custom_queries are skipped."""
        rag_params = RAGParameters(
            rag_enabled=True,
            database="test_db",
            rag_top_k=5,
            custom_queries=["valid query", "", "  ", "another valid"],
        )

        with patch.object(service, "_perform_rag_search") as mock_search:
            mock_search.return_value = [mock_result]

            service._perform_rag_search_with_custom_queries(
                project_dir="/test/dir",
                project_config=MagicMock(),
                message="ignored",
                rag_params=rag_params,
            )

            # Verify only valid queries were executed (2, not 4)
            assert mock_search.call_count == 2
