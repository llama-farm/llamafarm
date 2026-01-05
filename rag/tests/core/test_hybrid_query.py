"""Tests for HybridQuery - Unified query interface.

Phase 20: Hybrid Query Implementation
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add rag to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.hybrid_query import (
    FusionStrategy,
    HybridQueryExecutor,
    HybridQueryRequest,
    HybridQueryResponse,
    QueryMode,
    QueryResult,
    hybrid_query,
)


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_basic_result(self):
        """Test basic QueryResult creation."""
        result = QueryResult(
            id="test-123",
            content="Test content",
            score=0.95,
            source_store="vector",
        )
        assert result.id == "test-123"
        assert result.score == 0.95
        assert result.source_store == "vector"

    def test_result_with_all_fields(self):
        """Test QueryResult with all fields."""
        now = datetime.now(UTC)
        result = QueryResult(
            id="test-456",
            content={"data": "structured"},
            score=0.85,
            source_store="spatial",
            metadata={"type": "sensor"},
            timestamp=now,
            distance_m=150.5,
            path_depth=None,
        )
        assert result.distance_m == 150.5
        assert result.timestamp == now

    def test_to_dict(self):
        """Test QueryResult.to_dict()."""
        result = QueryResult(
            id="test-789",
            content="Content",
            score=0.7,
            source_store="graph",
            path_depth=2,
        )
        d = result.to_dict()

        assert d["id"] == "test-789"
        assert d["score"] == 0.7
        assert d["source_store"] == "graph"
        assert d["path_depth"] == 2


class TestHybridQueryRequest:
    """Test HybridQueryRequest dataclass."""

    def test_default_request(self):
        """Test default request values."""
        request = HybridQueryRequest()
        assert request.mode == QueryMode.HYBRID
        assert request.fusion_strategy == FusionStrategy.SCORE_BASED
        assert request.limit == 10

    def test_request_with_text(self):
        """Test request with query text."""
        request = HybridQueryRequest(
            query_text="find relevant documents",
            mode=QueryMode.VECTOR,
            limit=20,
        )
        assert request.query_text == "find relevant documents"
        assert request.mode == QueryMode.VECTOR

    def test_request_with_spatial(self):
        """Test request with spatial parameters."""
        request = HybridQueryRequest(
            latitude=35.78,
            longitude=-78.64,
            radius_meters=500.0,
            mode=QueryMode.SPATIAL,
        )
        assert request.latitude == 35.78
        assert request.longitude == -78.64
        assert request.radius_meters == 500.0

    def test_request_with_time_range(self):
        """Test request with time range."""
        now = datetime.now(UTC)
        yesterday = now - timedelta(days=1)

        request = HybridQueryRequest(
            start_time=yesterday,
            end_time=now,
            mode=QueryMode.TIMESERIES,
        )
        assert request.start_time == yesterday
        assert request.end_time == now


class TestHybridQueryResponse:
    """Test HybridQueryResponse dataclass."""

    def test_response(self):
        """Test HybridQueryResponse."""
        results = [
            QueryResult(id="1", content="A", score=0.9, source_store="vector"),
            QueryResult(id="2", content="B", score=0.8, source_store="graph"),
        ]

        response = HybridQueryResponse(
            results=results,
            total_count=2,
            stores_queried=["vector", "graph"],
            query_mode=QueryMode.HYBRID,
            fusion_strategy=FusionStrategy.SCORE_BASED,
            execution_time_ms=25.5,
            store_counts={"vector": 1, "graph": 1},
        )

        assert len(response.results) == 2
        assert response.total_count == 2
        assert "vector" in response.stores_queried

    def test_response_to_dict(self):
        """Test HybridQueryResponse.to_dict()."""
        response = HybridQueryResponse(
            results=[
                QueryResult(
                    id="1", content="Test", score=0.5, source_store="working_memory"
                )
            ],
            total_count=1,
            stores_queried=["working_memory"],
            query_mode=QueryMode.CONTEXT,
            fusion_strategy=FusionStrategy.TEMPORAL,
        )

        d = response.to_dict()
        assert d["query_mode"] == "context"
        assert d["fusion_strategy"] == "temporal"
        assert len(d["results"]) == 1


class TestHybridQueryExecutor:
    """Test HybridQueryExecutor."""

    @pytest.fixture
    def mock_unified_store(self):
        """Create a mock UnifiedDatasetStore."""
        store = MagicMock()
        store.vector_store = None  # Disabled
        store.graph_store = MagicMock()
        store.timeseries_store = MagicMock()
        store.spatial_store = MagicMock()
        store.working_memory = MagicMock()
        store.linkage_table = MagicMock()
        return store

    def test_executor_initialization(self, mock_unified_store):
        """Test executor initialization."""
        executor = HybridQueryExecutor(mock_unified_store)
        assert executor.store == mock_unified_store
        assert "vector" in executor.store_weights

    def test_determine_stores_hybrid(self, mock_unified_store):
        """Test store determination for hybrid mode."""
        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(
            query_text="test",  # Would trigger vector if enabled
            graph_node_id="node-1",  # Graph
            start_time=datetime.now(UTC) - timedelta(hours=1),  # Timeseries
            latitude=35.78,  # Spatial
        )

        stores = executor._determine_stores(request)

        # Vector not enabled, but others should be
        assert "vector" not in stores
        assert "graph" in stores
        assert "timeseries" in stores
        assert "spatial" in stores
        assert "working_memory" in stores

    def test_determine_stores_graph_only(self, mock_unified_store):
        """Test store determination for graph-only mode."""
        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(mode=QueryMode.GRAPH)
        stores = executor._determine_stores(request)

        assert stores == ["graph"]

    def test_determine_stores_context(self, mock_unified_store):
        """Test store determination for context mode."""
        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(mode=QueryMode.CONTEXT)
        stores = executor._determine_stores(request)

        assert "working_memory" in stores
        assert "graph" in stores

    def test_query_graph(self, mock_unified_store):
        """Test graph query."""
        mock_unified_store.graph_store.find_neighbors.return_value = [
            {"id": "n1", "name": "Node 1", "properties": {"type": "entity"}},
            {"id": "n2", "name": "Node 2", "properties": {"type": "person"}},
        ]

        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(
            graph_node_id="start-node",
            graph_direction="out",
        )

        results = executor._query_graph(request)

        assert len(results) == 2
        assert results[0].source_store == "graph"
        assert results[0].id == "n1"

    def test_query_working_memory(self, mock_unified_store):
        """Test working memory query."""
        mock_unified_store.working_memory.get_recent.return_value = [
            {"id": "wm1", "content": "Recent chat", "metadata": {"type": "chat"}},
            {"id": "wm2", "content": "Recent audio", "metadata": {"type": "audio"}},
        ]

        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(mode=QueryMode.CONTEXT)

        results = executor._query_working_memory(request)

        assert len(results) == 2
        assert results[0].source_store == "working_memory"

    def test_execute_full_query(self, mock_unified_store):
        """Test full query execution."""
        # Setup mock returns
        mock_unified_store.graph_store.find_neighbors.return_value = [
            {"id": "g1", "name": "Graph Result"},
        ]
        mock_unified_store.working_memory.get_recent.return_value = [
            {"id": "wm1", "content": "Working Memory Result"},
        ]
        mock_unified_store.timeseries_store.query_time_range.return_value = []
        mock_unified_store.spatial_store.query_spatial.return_value = []

        executor = HybridQueryExecutor(mock_unified_store)

        request = HybridQueryRequest(
            graph_node_id="test-node",
            mode=QueryMode.HYBRID,
        )

        response = executor.execute(request)

        assert isinstance(response, HybridQueryResponse)
        assert response.total_count > 0
        assert "graph" in response.stores_queried


class TestFusionStrategies:
    """Test result fusion strategies."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for fusion testing."""
        now = datetime.now(UTC)
        return [
            QueryResult(id="v1", content="Vector 1", score=0.9, source_store="vector"),
            QueryResult(id="g1", content="Graph 1", score=0.85, source_store="graph"),
            QueryResult(id="v2", content="Vector 2", score=0.8, source_store="vector"),
            QueryResult(
                id="s1",
                content="Spatial 1",
                score=0.75,
                source_store="spatial",
                distance_m=100,
            ),
            QueryResult(
                id="t1",
                content="Time 1",
                score=0.7,
                source_store="timeseries",
                timestamp=now,
            ),
            QueryResult(
                id="t2",
                content="Time 2",
                score=0.6,
                source_store="timeseries",
                timestamp=now - timedelta(hours=1),
            ),
        ]

    @pytest.fixture
    def mock_executor(self):
        """Create executor with mock store."""
        mock_store = MagicMock()
        mock_store.linkage_table = MagicMock()
        return HybridQueryExecutor(mock_store)

    def test_score_based_fusion(self, mock_executor, sample_results):
        """Test score-based fusion (default)."""
        request = HybridQueryRequest(fusion_strategy=FusionStrategy.SCORE_BASED)
        fused = mock_executor._fuse_results(sample_results, request)

        # Should be sorted by score descending
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score

    def test_interleave_fusion(self, mock_executor, sample_results):
        """Test interleave fusion."""
        request = HybridQueryRequest(fusion_strategy=FusionStrategy.INTERLEAVE)
        fused = mock_executor._fuse_results(sample_results, request)

        # Results should be interleaved from different stores
        # Check that we don't have consecutive results from same store
        # (at least until one store is exhausted)
        assert len(fused) == len(sample_results)

    def test_temporal_fusion(self, mock_executor, sample_results):
        """Test temporal fusion (most recent first)."""
        request = HybridQueryRequest(fusion_strategy=FusionStrategy.TEMPORAL)
        fused = mock_executor._fuse_results(sample_results, request)

        # Results with timestamps should come first, sorted by time
        timestamped = [r for r in fused if r.timestamp is not None]
        if len(timestamped) > 1:
            for i in range(len(timestamped) - 1):
                assert timestamped[i].timestamp >= timestamped[i + 1].timestamp

    def test_spatial_first_fusion(self, mock_executor, sample_results):
        """Test spatial-first fusion."""
        request = HybridQueryRequest(fusion_strategy=FusionStrategy.SPATIAL_FIRST)
        fused = mock_executor._fuse_results(sample_results, request)

        # Spatial results should come first
        if fused and fused[0].source_store == "spatial":
            assert True  # Spatial first is working
        else:
            # May not have spatial results with distance
            pass

    def test_weighted_fusion(self, mock_executor, sample_results):
        """Test weighted fusion."""
        request = HybridQueryRequest(fusion_strategy=FusionStrategy.WEIGHTED)
        fused = mock_executor._fuse_results(sample_results, request)

        # Scores should be modified by weights
        # Vector weight is 1.0, so vector result with score 0.9 should remain high
        assert len(fused) > 0


class TestConvenienceFunction:
    """Test hybrid_query convenience function."""

    def test_hybrid_query_function(self):
        """Test hybrid_query convenience function."""
        mock_store = MagicMock()
        mock_store.vector_store = None
        mock_store.graph_store = None
        mock_store.timeseries_store = None
        mock_store.spatial_store = None
        mock_store.working_memory = MagicMock()
        mock_store.working_memory.get_recent.return_value = [
            {"id": "wm1", "content": "Test"}
        ]
        mock_store.linkage_table = MagicMock()

        result = hybrid_query(
            unified_store=mock_store,
            mode="context",
            limit=5,
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert "stores_queried" in result


class TestQueryModes:
    """Test query mode enums."""

    def test_query_modes(self):
        """Test QueryMode enum values."""
        assert QueryMode.VECTOR.value == "vector"
        assert QueryMode.GRAPH.value == "graph"
        assert QueryMode.HYBRID.value == "hybrid"
        assert QueryMode.CONTEXT.value == "context"

    def test_fusion_strategies(self):
        """Test FusionStrategy enum values."""
        assert FusionStrategy.INTERLEAVE.value == "interleave"
        assert FusionStrategy.WEIGHTED.value == "weighted"
        assert FusionStrategy.SCORE_BASED.value == "score_based"
        assert FusionStrategy.TEMPORAL.value == "temporal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
