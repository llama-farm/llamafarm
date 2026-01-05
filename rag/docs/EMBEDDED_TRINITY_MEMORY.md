# Embedded Trinity Memory System

## Phase 3: Unified Dataset Architecture

The Embedded Trinity Memory System is LlamaFarm's brain-like memory architecture, providing federated embedded storage with cross-store linking, unified querying, and automatic consolidation.

## Overview

The system consists of multiple specialized stores, each optimized for different data types:

| Store | Technology | Purpose |
|-------|------------|---------|
| **Vector Store** | ChromaDB | Semantic search over embeddings |
| **Graph Store** | DuckDB | Entity relationships, knowledge graphs |
| **TimeSeries Store** | DuckDB | Time-based data, aggregations |
| **Spatial Store** | DuckDB + Spatial | Geo-location queries |
| **Working Memory** | DuckDB | Short-term buffer with TTL |
| **Linkage Table** | DuckDB | Cross-store UUID mapping |

## Dataset Types

The new Unified Dataset Architecture provides pre-configured dataset types:

| Dataset Type | Vector | Graph | TimeSeries | Spatial | Working Memory |
|-------------|--------|-------|------------|---------|----------------|
| `knowledge` | ✓ | ✓ | - | - | - |
| `realtime` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `graph` | - | ✓ | - | - | - |
| `timeseries` | - | - | ✓ | - | ✓ |
| `spatial` | - | - | - | ✓ | ✓ |
| `hybrid` | ✓ | ✓ | ✓ | ✓ | ✓ |

## Quick Start

### Using UnifiedDatasetStore

```python
from core.unified_store import UnifiedDatasetStore

# Create a knowledge dataset (vector + graph)
store = UnifiedDatasetStore(
    dataset_config={"name": "my_knowledge", "type": "knowledge"},
    project_dir="/path/to/project",
)

# Add a document with entity extraction
result = store.add_document(
    content="John Smith works at Apple Inc in San Francisco.",
    doc_id="doc-1",
    extract_entities=True,
)
print(f"Document stored in: {result['stores']}")
print(f"Entities extracted: {result.get('entities', 0)}")

# Query the knowledge graph
neighbors = store.query(
    query_type="graph",
    graph_query={"node_id": "person:john_smith", "direction": "outgoing"}
)

store.close()
```

### Using Realtime Dataset for Streaming

```python
from core.unified_store import UnifiedDatasetStore

# Create a realtime dataset (all stores enabled)
store = UnifiedDatasetStore(
    dataset_config={"name": "iot_telemetry", "type": "realtime"},
    project_dir="/path/to/project",
)

# Stream telemetry data
store.add_stream_record(
    data={"temperature": 72.5, "humidity": 45},
    data_type="sensor",
    latitude=35.78,
    longitude=-78.64,
    metadata={"sensor_id": "sensor-001"},
)

# Query by time range
results = store.query(
    query_type="timeseries",
    time_range={"start": one_hour_ago, "end": now},
)

# Query by location
results = store.query(
    query_type="spatial",
    spatial={"latitude": 35.78, "longitude": -78.64, "radius_meters": 500},
)

# Get recent working memory context
results = store.query(query_type="recent")

store.close()
```

## Configuration in llamafarm.yaml

### Typed Datasets

```yaml
# llamafarm.yaml
version: v1
name: my_project

# Dataset definitions with type system
datasets:
  # Knowledge dataset for document RAG
  - name: documents
    type: knowledge
    data_processing_strategy: universal_processor
    database: main_database
    graph:
      entity_extraction: true
      max_path_depth: 10

  # Realtime dataset for IoT/streaming
  - name: telemetry
    type: realtime
    timeseries:
      retention_days: 30
    spatial:
      index_type: rtree
    working_memory:
      ttl_seconds: 3600
      max_records: 10000

  # Graph-only dataset for relationships
  - name: entity_graph
    type: graph
    graph:
      max_path_depth: 15
```

### Memory Store Configuration

```yaml
# Memory stores (legacy, still supported)
memory:
  default_store: trinity_memory
  stores:
    - name: trinity_memory
      working_memory:
        ttl_seconds: 3600
        max_records: 10000
      timeseries:
        retention_days: 30
      graph:
        max_path_depth: 10
      consolidation:
        min_records: 10
        batch_size: 100
        prune_after_consolidate: true
```

## Hybrid Query System

The `HybridQueryExecutor` provides intelligent querying across multiple stores:

```python
from core.hybrid_query import HybridQueryExecutor, HybridQueryRequest, QueryMode, FusionStrategy

executor = HybridQueryExecutor(store)

# Multi-store query with score-based fusion
request = HybridQueryRequest(
    query_text="emergency medical",        # For vector search
    graph_node_id="team:alpha",            # For graph traversal
    start_time=one_hour_ago,               # For timeseries filter
    latitude=35.78, longitude=-78.64,      # For spatial filter
    mode=QueryMode.HYBRID,
    fusion_strategy=FusionStrategy.SCORE_BASED,
    limit=10,
)

response = executor.execute(request)
print(f"Total results: {response.total_count}")
print(f"Stores queried: {response.stores_queried}")
print(f"Execution time: {response.execution_time_ms}ms")
```

### Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `INTERLEAVE` | Round-robin from each store |
| `WEIGHTED` | Weight-based ranking by store type |
| `SCORE_BASED` | Rank purely by relevance score |
| `TEMPORAL` | Most recent first |
| `SPATIAL_FIRST` | Closest locations first, then others |

### Query Caching (Phase 26)

Query results are automatically cached for repeated queries:

```python
executor = HybridQueryExecutor(
    store,
    enable_cache=True,
    cache_max_size=100,
    cache_ttl_seconds=30,
)

# First query (cache miss)
response1 = executor.execute(request)
print(f"Cache hit: {response1.metadata.get('cache_hit')}")  # False

# Second identical query (cache hit)
response2 = executor.execute(request)
print(f"Cache hit: {response2.metadata.get('cache_hit')}")  # True

# Check cache statistics
stats = executor.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

## Entity Extraction Pipeline

The `EntityExtractor` automatically extracts named entities from documents:

```python
from components.extractors.entity_extractor import EntityExtractor
from core.base import Document

extractor = EntityExtractor(
    name="MyExtractor",
    config={
        "entity_types": ["PERSON", "ORG", "GPE", "DATE"],
        "use_fallback": True,  # Use regex if spaCy unavailable
    },
)

doc = Document(
    id="doc-1",
    content="John Smith works at Apple Inc in San Francisco.",
)

entities = extractor.extract_entities(doc)
for entity in entities:
    print(f"{entity.name} ({entity.entity_type}): {entity.confidence:.2f}")
```

### Extracting Entities to Graph

```python
# Extract entities and add to graph store
result = extractor.extract_to_graph(
    document=doc,
    graph_store=store.graph_store,
    linkage_table=store.linkage_table,
)
print(f"Nodes created: {result['nodes_created']}")
print(f"Edges created: {result['edges_created']}")
```

## Consolidation (Memory Synthesis)

The `Consolidator` acts as the "hippocampus" - synthesizing facts from working memory:

```python
from core.consolidator import Consolidator

consolidator = Consolidator(
    memory_store=store,
    config={
        "buffer_threshold": 10,
        "use_entity_extractor": True,  # Use NER for fact extraction
    },
)

# Run consolidation cycle
result = consolidator.run_cycle(use_llm=False)  # Rule-based
print(f"Records processed: {result['records_processed']}")
print(f"Facts extracted: {result['facts_extracted']}")
print(f"Nodes created: {result['nodes_created']}")
```

### Consolidator with UnifiedDatasetStore

The consolidator automatically detects and adapts to the store type:

```python
# Works with both MemoryStore and UnifiedDatasetStore
consolidator = Consolidator(memory_store=unified_store)

# Automatically uses the correct methods:
# - unified_store.add_node() for graph
# - unified_store.working_memory.prune() for cleanup
```

## RAG Pipeline Integration

Use `DatasetIntegratedPipeline` to process documents through the full RAG pipeline:

```python
from core.pipeline_integration import DatasetIntegratedPipeline, process_documents_to_dataset
from core.base import Document

# Create integrated pipeline
pipeline = DatasetIntegratedPipeline(
    name="Knowledge Pipeline",
    dataset_store=store,
    config={
        "extract_entities": True,
        "extract_relationships": True,
    },
)

# Process documents
documents = [
    Document(id="doc-1", content="Alice works at Acme Corp..."),
    Document(id="doc-2", content="Bob manages the engineering team..."),
]

result = pipeline.process_with_dataset(
    documents=documents,
    store_in_vector=True,
    store_in_graph=True,
)

print(f"Processed: {len(result.documents)} documents")
print(f"Errors: {len(result.errors)}")
```

### Convenience Function

```python
result = process_documents_to_dataset(
    documents=documents,
    project_path="/path/to/project",
    dataset_name="my_knowledge",
    dataset_type="knowledge",
    extract_entities=True,
)
```

## Performance Features (Phase 26)

### Connection Pooling

DuckDB stores support connection pooling for concurrent access:

```python
from components.stores.duckdb_store import DuckDBStore

store = DuckDBStore(config={
    "path": "/path/to/db.duckdb",
    "use_pool": True,
    "pool_size": 5,
})

# Connections are automatically managed
store.add_records(records)  # Uses pooled connection
store.query_time_range(...)  # Uses pooled connection
```

### Batch Inserts

Large record sets are automatically batch-inserted:

```python
store = DuckDBStore(config={
    "path": "/path/to/db.duckdb",
    "batch_size": 1000,  # Batch insert threshold
})

# 5000 records will be inserted in 5 batches
records = [{"source": "sensor", "data": {"value": i}} for i in range(5000)]
inserted = store.add_records(records)
```

## Cross-Store Linking

The `LinkageTable` maintains UUID mappings across all stores:

```python
# Create a linked concept
store.linkage_table.link(
    concept_uuid="rescue_event_001",
    vector_id="chroma_doc_001",
    graph_node_id="node_rescue_001",
    timeseries_row_id="ts_batch_001",
)

# Find concept from any component ID
links = store.linkage_table.get_links("rescue_event_001")
print(f"Graph node: {links['graph_node_id']}")

# Cascade delete - get all IDs for cleanup
ids = store.linkage_table.unlink("rescue_event_001")
```

## Statistics

Get comprehensive statistics from all stores:

```python
stats = store.get_stats()
print(f"Dataset: {stats['dataset_name']} ({stats['dataset_type']})")
print(f"Enabled stores: {stats['enabled_stores']}")

if "graph" in stats["stores"]:
    print(f"Graph nodes: {stats['stores']['graph']['node_count']}")
    print(f"Graph edges: {stats['stores']['graph']['edge_count']}")

if "timeseries" in stats["stores"]:
    print(f"Time-series records: {stats['stores']['timeseries']['record_count']}")

if "working_memory" in stats["stores"]:
    print(f"Working memory records: {stats['stores']['working_memory']['total_records']}")
```

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        UnifiedDatasetStore                                │
│                   (Dataset Type: knowledge/realtime/hybrid)               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  ChromaDB     │    │     DuckDB        │    │     DuckDB        │
│  VectorStore  │    │   TimeSeriesStore │    │    GraphStore     │
│  (embeddings) │    │   (telemetry)     │    │   (entities)      │
└───────────────┘    └───────────────────┘    └───────────────────┘
        │                        │                        │
        │            ┌───────────┴───────────┐            │
        │            │                       │            │
        │            ▼                       ▼            │
        │    ┌───────────────┐    ┌───────────────────┐   │
        │    │    DuckDB     │    │      DuckDB       │   │
        │    │ SpatialStore  │    │  WorkingMemory    │   │
        │    │  (geo-queries)│    │   (TTL buffer)    │   │
        │    └───────────────┘    └───────────────────┘   │
        │                                                  │
        └──────────────────────┬──────────────────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │   LinkageTable    │
                    │ UUID → Store IDs  │
                    └───────────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │  HybridQuery      │
                    │  Executor         │
                    │  (with caching)   │
                    └───────────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │   Consolidator    │
                    │  ("hippocampus")  │
                    └───────────────────┘
```

## API Reference

### UnifiedDatasetStore

| Method | Description |
|--------|-------------|
| `add_document(content, doc_id, metadata, extract_entities)` | Add document to vector + graph |
| `add_stream_record(data, data_type, timestamp, lat, lon, metadata)` | Add streaming data |
| `add_node(name, node_type, node_id, properties)` | Add node to graph |
| `add_edge(source_id, target_id, relationship, weight, properties)` | Add edge to graph |
| `query(query_type, time_range, spatial, graph_query, limit)` | Unified query |
| `get_enabled_stores()` | List enabled store names |
| `get_stats()` | Get statistics from all stores |
| `clear()` | Clear all store data |
| `close()` | Close all connections |

### HybridQueryExecutor

| Method | Description |
|--------|-------------|
| `execute(request)` | Execute hybrid query |
| `get_cache_stats()` | Get cache statistics |
| `clear_cache()` | Clear query cache |

### Consolidator

| Method | Description |
|--------|-------------|
| `get_pending_records(limit)` | Get records to consolidate |
| `run_cycle(use_llm)` | Run consolidation cycle |
| `prune()` | Prune expired working memory |

## Examples

See the `examples/database/` directory for complete demos:

- `demo_unified_dataset.py` - Unified dataset operations
- `demo_hybrid_query.py` - Multi-store hybrid queries
- `demo_consolidator.py` - Memory consolidation
- `demo_entity_extraction.py` - Entity extraction pipeline

Run all demos:
```bash
cd rag
bash ../examples/database/run_all_demos.sh
```

## Testing

Run the test suite:
```bash
cd rag
uv run pytest tests/ -v
```

Run specific test modules:
```bash
# Core tests
uv run pytest tests/core/ -v

# E2E integration tests
uv run pytest tests/e2e/ -v

# Performance tests
uv run pytest tests/core/test_performance.py -v
```
