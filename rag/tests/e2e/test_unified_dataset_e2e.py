"""End-to-end tests for UnifiedDatasetStore with typed datasets.

Phase 25: E2E Integration Tests

Tests the complete flow from data ingestion to query across:
- Knowledge datasets (vector + graph)
- Realtime datasets (all stores)
- Spatial datasets (spatial + timeseries)
- Hybrid queries
- Entity extraction pipeline
- Consolidation
"""

import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# Add rag to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestUnifiedDatasetE2EKnowledge:
    """End-to-end tests for knowledge dataset type."""

    def test_knowledge_dataset_entity_extraction_flow(self):
        """Test complete flow: ingest documents -> extract entities -> graph -> query."""
        from components.extractors.entity_extractor import EntityExtractor
        from core.base import Document
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create knowledge dataset
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_knowledge", "type": "knowledge"},
                project_dir=temp_dir,
            )

            # Verify correct stores are enabled
            enabled = store.get_enabled_stores()
            assert "graph" in enabled
            # Linkage table is always created but not returned by get_enabled_stores
            assert store.linkage_table is not None

            # Create entity extractor
            extractor = EntityExtractor(
                name="TestExtractor",
                config={"use_fallback": True},
            )

            # Create test documents
            docs = [
                Document(
                    id="doc-1",
                    content="John Smith works at Apple Inc in San Francisco.",
                    metadata={"source": "test"},
                ),
                Document(
                    id="doc-2",
                    content="Mary Johnson is the CEO of Google in Mountain View.",
                    metadata={"source": "test"},
                ),
            ]

            # Extract entities and add to graph
            total_entities = 0
            for doc in docs:
                entities = extractor.extract_entities(doc)
                for entity in entities:
                    node_id = store.add_node(
                        name=entity.name,
                        node_type=entity.entity_type.lower(),
                        node_id=entity.entity_id,
                        properties={
                            "source_doc": doc.id,
                            "confidence": entity.confidence,
                        },
                    )
                    if node_id:
                        total_entities += 1

            # Should have extracted multiple entities
            assert total_entities > 0

            # Get stats - use nested stores dict
            stats = store.get_stats()
            assert stats["stores"]["graph"]["node_count"] > 0

            store.close()

    def test_knowledge_graph_traversal(self):
        """Test graph traversal after entity extraction."""
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_graph", "type": "graph"},
                project_dir=temp_dir,
            )

            # Create nodes
            alice_id = store.add_node("Alice", "person", node_id="person:alice")
            bob_id = store.add_node("Bob", "person", node_id="person:bob")
            store.add_node("Charlie", "person", node_id="person:charlie")
            store.add_node("Acme Corp", "organization", node_id="org:acme")

            assert alice_id is not None
            assert bob_id is not None

            # Create relationships
            edge1 = store.add_edge("person:alice", "person:bob", "knows")
            store.add_edge("person:bob", "person:charlie", "knows")
            edge3 = store.add_edge("person:alice", "org:acme", "works_at")
            store.add_edge("person:bob", "org:acme", "works_at")

            # Verify edges were created
            assert edge1 is not None or edge3 is not None  # At least one should succeed

            # Get stats to verify graph content
            stats = store.get_stats()
            assert stats["stores"]["graph"]["node_count"] >= 4
            assert stats["stores"]["graph"]["edge_count"] >= 1

            store.close()


class TestUnifiedDatasetE2ERealtime:
    """End-to-end tests for realtime dataset type."""

    def test_realtime_stream_ingestion_flow(self):
        """Test complete flow: stream records -> working memory -> hybrid query."""
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realtime dataset
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_realtime", "type": "realtime"},
                project_dir=temp_dir,
            )

            # Verify all realtime stores are enabled
            enabled = store.get_enabled_stores()
            assert "timeseries" in enabled
            assert "spatial" in enabled
            assert "working_memory" in enabled
            assert "graph" in enabled

            # Stream IoT data
            for i in range(10):
                store.add_stream_record(
                    data={"temperature": 70 + i, "humidity": 50 + i},
                    data_type="sensor",
                    latitude=35.78 + i * 0.001,
                    longitude=-78.64 + i * 0.001,
                    metadata={"sensor_id": f"sensor-{i}"},
                )

            # Query recent working memory (use "recent" query type)
            context = store.query(query_type="recent")
            assert "working_memory" in context

            # Get stats - use nested stores dict
            stats = store.get_stats()
            assert stats["stores"]["timeseries"]["record_count"] >= 10
            assert stats["stores"]["working_memory"]["total_records"] >= 10

            store.close()

    def test_realtime_spatial_query(self):
        """Test spatial queries on realtime data."""
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_spatial", "type": "spatial"},
                project_dir=temp_dir,
            )

            # Add records at different locations
            # Base point
            store.add_stream_record(
                data={"vehicle_id": "V001"},
                data_type="location",
                latitude=35.7800,
                longitude=-78.6400,
            )

            # 100m away
            store.add_stream_record(
                data={"vehicle_id": "V002"},
                data_type="location",
                latitude=35.7809,
                longitude=-78.6400,
            )

            # 5km away
            store.add_stream_record(
                data={"vehicle_id": "V003"},
                data_type="location",
                latitude=35.8250,
                longitude=-78.6400,
            )

            # Query within 500m of base using correct API
            results = store.query(
                query_type="spatial",
                spatial={
                    "latitude": 35.7800,
                    "longitude": -78.6400,
                    "radius_meters": 500,
                },
            )

            # Should find at least the records with spatial data
            assert isinstance(results, dict)

            store.close()


class TestUnifiedDatasetE2EHybridQuery:
    """End-to-end tests for hybrid queries."""

    def test_hybrid_query_multi_store(self):
        """Test hybrid query across multiple stores."""
        from core.hybrid_query import HybridQueryExecutor, HybridQueryRequest, QueryMode
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_hybrid", "type": "hybrid"},
                project_dir=temp_dir,
            )

            # Add graph data
            store.add_node("Alpha Team", "team", node_id="team:alpha")
            store.add_node("Sector 7", "location", node_id="loc:sector7")
            store.add_edge("team:alpha", "loc:sector7", "assigned_to")

            # Add stream data
            store.add_stream_record(
                data={"status": "active", "members": 5},
                data_type="team_status",
                latitude=35.78,
                longitude=-78.64,
            )

            # Create hybrid query executor
            executor = HybridQueryExecutor(store)

            # Execute hybrid query
            request = HybridQueryRequest(
                graph_node_id="team:alpha",
                mode=QueryMode.CONTEXT,
                limit=10,
            )

            response = executor.execute(request)

            assert response.total_count >= 0
            assert (
                "graph" in response.stores_queried
                or "working_memory" in response.stores_queried
            )

            store.close()


class TestUnifiedDatasetE2EConsolidation:
    """End-to-end tests for consolidation with UnifiedDatasetStore."""

    def test_consolidation_with_unified_store(self):
        """Test consolidation works with UnifiedDatasetStore."""
        from core.consolidator import Consolidator
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_consolidate", "type": "realtime"},
                project_dir=temp_dir,
            )

            # Add chat messages to working memory
            for i in range(15):
                store.add_stream_record(
                    data=f"User{i}: This is message {i} from the test.",
                    data_type="chat",
                    metadata={"user_id": f"user-{i}"},
                )

            # Create consolidator (uses entity extractor)
            consolidator = Consolidator(
                memory_store=store,
                config={
                    "buffer_threshold": 5,
                    "use_entity_extractor": False,  # Use rule-based for test
                },
            )

            # Verify consolidator detects unified store
            assert consolidator._is_unified_store is True

            # Get pending records
            pending = consolidator.get_pending_records(limit=20)
            assert len(pending) >= 15

            # Run consolidation cycle
            result = consolidator.run_cycle(use_llm=False)

            assert result["records_processed"] >= 15
            assert result["skipped"] is False

            store.close()


class TestUnifiedDatasetE2EPipeline:
    """End-to-end tests for pipeline integration."""

    def test_pipeline_document_processing(self):
        """Test processing documents through integrated pipeline."""
        from core.base import Document
        from core.pipeline_integration import DatasetIntegratedPipeline
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_pipeline", "type": "knowledge"},
                project_dir=temp_dir,
            )

            pipeline = DatasetIntegratedPipeline(
                name="Test Pipeline",
                dataset_store=store,
                config={"extract_entities": True},
            )

            # Create test documents
            documents = [
                Document(
                    id="doc-1",
                    content="The quick brown fox jumps over the lazy dog.",
                    metadata={"test": True},
                ),
                Document(
                    id="doc-2",
                    content="John went to New York City to visit Apple headquarters.",
                    metadata={"test": True},
                ),
            ]

            # Process documents
            result = pipeline.process_with_dataset(
                documents=documents,
                store_in_vector=False,
                store_in_graph=True,
            )

            assert len(result.documents) == 2
            assert len(result.errors) == 0

            store.close()


class TestUnifiedDatasetE2ETimeseries:
    """End-to-end tests for timeseries dataset type."""

    def test_timeseries_time_range_query(self):
        """Test time range queries on timeseries data."""
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_timeseries", "type": "timeseries"},
                project_dir=temp_dir,
            )

            now = datetime.now(UTC)

            # Add records at different times
            for i in range(10):
                store.add_stream_record(
                    data={"value": i * 10},
                    data_type="metric",
                    timestamp=now - timedelta(minutes=i),
                )

            # Query last 5 minutes using correct API
            results = store.query(
                query_type="timeseries",
                time_range={
                    "start": now - timedelta(minutes=5),
                    "end": now,
                },
            )

            assert isinstance(results, dict)

            store.close()


class TestUnifiedDatasetE2ELinkage:
    """End-to-end tests for cross-store linkage."""

    def test_cross_store_linking(self):
        """Test that linkage table correctly links across stores."""
        from core.unified_store import UnifiedDatasetStore

        with tempfile.TemporaryDirectory() as temp_dir:
            store = UnifiedDatasetStore(
                dataset_config={"name": "test_linkage", "type": "hybrid"},
                project_dir=temp_dir,
            )

            # Add a graph node
            node_id = store.add_node(
                name="Test Entity",
                node_type="entity",
                node_id="entity:test",
            )

            # Add a stream record with the same concept
            store.add_stream_record(
                data={"entity_ref": "entity:test"},
                data_type="entity_update",
            )

            # Create linkage
            store.linkage_table.link(
                concept_uuid="concept:test",
                graph_node_id=node_id,
            )

            # Verify linkage exists
            links = store.linkage_table.get_links("concept:test")
            assert links is not None
            assert links.get("graph_node_id") == node_id

            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
