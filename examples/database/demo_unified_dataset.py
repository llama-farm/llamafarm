#!/usr/bin/env python3
"""
Unified Dataset Store Demo - Phase 3 Unified Dataset Architecture

This demo showcases the new UnifiedDatasetStore that provides:
1. Typed datasets (knowledge, realtime, graph, timeseries, spatial, hybrid)
2. Automatic store selection based on dataset type
3. Entity extraction and graph population
4. Hybrid querying across all stores
5. Query result caching

Run from the rag directory:
    cd rag && uv run python ../examples/database/demo_unified_dataset.py
"""

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add rag to path for direct component access
sys.path.insert(0, ".")


def print_header(title: str) -> None:
    """Print a fancy header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"  ✓ {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"  → {msg}")


def print_data(label: str, value) -> None:
    """Print data with label."""
    print(f"    {label}: {value}")


def main() -> int:
    """Run the unified dataset demo."""

    print_header("UNIFIED DATASET STORE DEMO")
    print("  Demonstrating the Phase 3 Unified Dataset Architecture")

    from core.unified_store import UnifiedDatasetStore
    from core.hybrid_query import HybridQueryExecutor, HybridQueryRequest, QueryMode

    with tempfile.TemporaryDirectory(prefix="unified_demo_") as temp_dir:
        print_info(f"Demo data directory: {temp_dir}")

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1: Knowledge Dataset
        # ═══════════════════════════════════════════════════════════════════
        print_section("1. KNOWLEDGE DATASET (vector + graph)")

        knowledge_store = UnifiedDatasetStore(
            dataset_config={"name": "knowledge_demo", "type": "knowledge"},
            project_dir=temp_dir,
        )

        print_success("Created knowledge dataset")
        print_data("Enabled stores", knowledge_store.get_enabled_stores())

        # Add graph nodes (entities)
        print_info("Adding entities to knowledge graph...")
        knowledge_store.add_node("Alice Johnson", "person", node_id="person:alice")
        knowledge_store.add_node("Bob Smith", "person", node_id="person:bob")
        knowledge_store.add_node("Acme Corp", "organization", node_id="org:acme")
        knowledge_store.add_node("Tech Startup", "organization", node_id="org:startup")

        # Add relationships
        knowledge_store.add_edge("person:alice", "org:acme", "works_at")
        knowledge_store.add_edge("person:bob", "org:acme", "works_at")
        knowledge_store.add_edge("person:alice", "person:bob", "manages")
        knowledge_store.add_edge("org:startup", "org:acme", "acquired_by")

        print_success("Added 4 entities and 4 relationships")

        # Get stats
        stats = knowledge_store.get_stats()
        print_data("Graph nodes", stats["stores"]["graph"]["node_count"])
        print_data("Graph edges", stats["stores"]["graph"]["edge_count"])

        knowledge_store.close()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2: Realtime Dataset
        # ═══════════════════════════════════════════════════════════════════
        print_section("2. REALTIME DATASET (all stores enabled)")

        realtime_store = UnifiedDatasetStore(
            dataset_config={"name": "realtime_demo", "type": "realtime"},
            project_dir=temp_dir,
        )

        print_success("Created realtime dataset")
        print_data("Enabled stores", realtime_store.get_enabled_stores())

        # Stream telemetry data
        print_info("Streaming IoT telemetry...")
        now = datetime.now(timezone.utc)
        for i in range(20):
            realtime_store.add_stream_record(
                data={"temperature": 70 + i * 0.5, "humidity": 45 + i % 10},
                data_type="sensor",
                timestamp=now - timedelta(minutes=20 - i),
                latitude=35.78 + i * 0.001,
                longitude=-78.64 + i * 0.001,
                metadata={"sensor_id": f"sensor-{i % 3}"},
            )
        print_success("Streamed 20 telemetry records")

        # Add some entities to the graph
        realtime_store.add_node("Sensor Hub", "device", node_id="device:hub")
        realtime_store.add_node("Warehouse A", "location", node_id="loc:warehouse_a")
        realtime_store.add_edge("device:hub", "loc:warehouse_a", "located_at")

        stats = realtime_store.get_stats()
        print_data("TimeSeries records", stats["stores"]["timeseries"]["record_count"])
        print_data("Working memory records", stats["stores"]["working_memory"]["total_records"])
        print_data("Graph nodes", stats["stores"]["graph"]["node_count"])

        realtime_store.close()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3: Spatial Dataset
        # ═══════════════════════════════════════════════════════════════════
        print_section("3. SPATIAL DATASET (spatial + working memory)")

        spatial_store = UnifiedDatasetStore(
            dataset_config={"name": "spatial_demo", "type": "spatial"},
            project_dir=temp_dir,
        )

        print_success("Created spatial dataset")
        print_data("Enabled stores", spatial_store.get_enabled_stores())

        # Add location data
        print_info("Adding location data...")
        locations = [
            (35.7796, -78.6382, "Checkpoint Alpha"),
            (35.7850, -78.6400, "Checkpoint Beta"),
            (35.7880, -78.6420, "Rescue Zone"),
            (35.7700, -78.6300, "Base Camp"),
            (35.7820, -78.6350, "Supply Depot"),
        ]

        for lat, lon, name in locations:
            spatial_store.add_stream_record(
                data={"name": name, "status": "active"},
                data_type="location",
                latitude=lat,
                longitude=lon,
            )
        print_success(f"Added {len(locations)} location records")

        # Spatial query
        print_info("Querying locations within 2km of Base Camp...")
        results = spatial_store.query(
            query_type="spatial",
            spatial={"latitude": 35.7700, "longitude": -78.6300, "radius_meters": 2000},
        )
        print_success(f"Found spatial results")

        spatial_store.close()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 4: Hybrid Dataset with Full Querying
        # ═══════════════════════════════════════════════════════════════════
        print_section("4. HYBRID DATASET WITH HYBRID QUERY")

        hybrid_store = UnifiedDatasetStore(
            dataset_config={"name": "hybrid_demo", "type": "hybrid"},
            project_dir=temp_dir,
        )

        print_success("Created hybrid dataset")
        print_data("Enabled stores", hybrid_store.get_enabled_stores())

        # Add data to all stores
        print_info("Populating all stores...")

        # Graph data
        hybrid_store.add_node("Team Alpha", "team", node_id="team:alpha")
        hybrid_store.add_node("Sector 7", "location", node_id="loc:sector7")
        hybrid_store.add_edge("team:alpha", "loc:sector7", "assigned_to")

        # Streaming data with spatial
        for i in range(10):
            hybrid_store.add_stream_record(
                data={"status": "active", "alert_level": i % 4},
                data_type="team_status",
                timestamp=now - timedelta(minutes=i),
                latitude=35.78 + i * 0.001,
                longitude=-78.64 + i * 0.001,
            )

        print_success("Added data to graph, timeseries, spatial, and working memory")

        # Create hybrid query executor with caching
        print_info("Creating HybridQueryExecutor with caching...")
        executor = HybridQueryExecutor(
            hybrid_store,
            enable_cache=True,
            cache_max_size=100,
            cache_ttl_seconds=60,
        )

        # Execute hybrid query
        print_info("Executing hybrid query across all stores...")
        request = HybridQueryRequest(
            graph_node_id="team:alpha",
            start_time=now - timedelta(hours=1),
            end_time=now,
            mode=QueryMode.HYBRID,
            limit=10,
        )

        response = executor.execute(request)
        print_success(f"Query returned {response.total_count} results")
        print_data("Stores queried", response.stores_queried)
        print_data("Execution time", f"{response.execution_time_ms:.2f}ms")
        print_data("Cache hit", response.metadata.get("cache_hit", False))

        # Execute same query again (should hit cache)
        print_info("Executing same query again (should hit cache)...")
        response2 = executor.execute(request)
        print_data("Cache hit", response2.metadata.get("cache_hit", False))
        print_data("Execution time", f"{response2.execution_time_ms:.2f}ms")

        # Cache stats
        cache_stats = executor.get_cache_stats()
        print_data("Cache hit rate", f"{cache_stats['hit_rate']:.1%}")

        stats = hybrid_store.get_stats()
        print_section("Final Statistics")
        print_data("Dataset name", stats["dataset_name"])
        print_data("Dataset type", stats["dataset_type"])
        print_data("Graph nodes", stats["stores"]["graph"]["node_count"])
        print_data("Graph edges", stats["stores"]["graph"]["edge_count"])
        print_data("TimeSeries records", stats["stores"]["timeseries"]["record_count"])
        print_data("Working memory", stats["stores"]["working_memory"]["total_records"])

        hybrid_store.close()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 5: Entity Extraction Pipeline
        # ═══════════════════════════════════════════════════════════════════
        print_section("5. ENTITY EXTRACTION PIPELINE")

        from core.base import Document
        from components.extractors.entity_extractor import EntityExtractor

        knowledge_store2 = UnifiedDatasetStore(
            dataset_config={"name": "entity_demo", "type": "knowledge"},
            project_dir=temp_dir,
        )

        extractor = EntityExtractor(
            name="DemoExtractor",
            config={"use_fallback": True},  # Use regex fallback
        )

        # Create documents
        documents = [
            Document(
                id="doc-1",
                content="John Smith works at Apple Inc in San Francisco. He joined in January 2020.",
                metadata={"source": "test"},
            ),
            Document(
                id="doc-2",
                content="Mary Johnson is the CEO of Google in Mountain View. The company was founded in 1998.",
                metadata={"source": "test"},
            ),
        ]

        print_info("Extracting entities from documents...")
        total_entities = 0
        for doc in documents:
            entities = extractor.extract_entities(doc)
            print(f"    {doc.id}: Found {len(entities)} entities")
            for entity in entities:
                node_id = knowledge_store2.add_node(
                    name=entity.name,
                    node_type=entity.entity_type.lower(),
                    node_id=entity.entity_id,
                    properties={"source_doc": doc.id, "confidence": entity.confidence},
                )
                if node_id:
                    total_entities += 1

        print_success(f"Extracted and stored {total_entities} entities")

        stats = knowledge_store2.get_stats()
        print_data("Total graph nodes", stats["stores"]["graph"]["node_count"])

        knowledge_store2.close()

    # Final summary
    print_header("DEMO COMPLETE")
    print("""
  The Unified Dataset Store demo showed:

  1. DATASET TYPES
     - knowledge: vector + graph for document RAG
     - realtime: all stores for streaming IoT
     - spatial: geo-location + working memory
     - hybrid: all stores with full hybrid query

  2. HYBRID QUERY EXECUTOR
     - Multi-store query routing
     - Result fusion with score-based ranking
     - Query result caching with TTL

  3. ENTITY EXTRACTION PIPELINE
     - Extract named entities from documents
     - Auto-populate knowledge graph
     - Cross-store linking via LinkageTable

  4. PERFORMANCE FEATURES
     - Query caching with LRU eviction
     - Connection pooling (optional)
     - Batch inserts for high-volume streaming

  See docs/EMBEDDED_TRINITY_MEMORY.md for full documentation!
""")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from the 'rag' directory: cd rag && uv run python ../examples/database/demo_unified_dataset.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
