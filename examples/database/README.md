# Database Examples

Examples demonstrating the Embedded Trinity Memory System - a federated embedded architecture for LlamaFarm's brain-like memory capabilities.

## Overview

The Embedded Trinity consists of:
- **Vector Memory** (ChromaDB) - Semantic search over unstructured text
- **Time-Series Memory** (DuckDB) - Telemetry, spatial queries, fast aggregations
- **Graph Memory** (DuckDB) - Entity relationships and knowledge graph
- **Working Memory** (DuckDB) - Short-term buffer with TTL expiration
- **Linkage Table** (DuckDB) - Cross-database UUID mapping for consistency

## Examples

| Example | Script | Description |
|---------|--------|-------------|
| DuckDB Store | `demo_duckdb_store.py` | Time-series storage with spatial queries |
| Graph Store | `demo_graph_store.py` | Entity relationships and path finding |
| Working Memory | `demo_working_memory.py` | Short-term buffer with TTL-based expiration |
| Linkage Table | `demo_linkage_table.py` | Cross-database record linking |
| MemoryStore | `demo_memory_store.py` | Unified interface for all stores |
| Consolidator | `demo_consolidator.py` | Memory synthesis agent (the "hippocampus") |
| Memory API | `demo_memory_api.py` | REST API client for memory operations |

## Running Examples

All examples are self-contained Python scripts that can be run directly:

```bash
# From the llamafarm root directory
cd rag

# Run individual demos
uv run python ../examples/database/demo_linkage_table.py
uv run python ../examples/database/demo_graph_store.py
uv run python ../examples/database/demo_working_memory.py
uv run python ../examples/database/demo_duckdb_store.py
```

Or run all demos:
```bash
bash ../examples/database/run_all_demos.sh
```

## Key Concepts

### Linkage Table - Cross-Database Consistency

The LinkageTable maps a single concept UUID to IDs in each store:

```python
from components.stores.duckdb_store import LinkageTable

table = LinkageTable(config={"path": "linkage.duckdb"})

# Create linked record across all stores
concept_id = table.link(
    concept_uuid="rescue_event_001",
    vector_id="chroma_doc_001",        # ChromaDB document
    graph_node_id="node_rescue_001",   # Graph node
    timeseries_row_id="ts_batch_001",  # Timeseries record
)

# Find concept from any component ID
uuid = table.find_by_any_id(vector_id="chroma_doc_001")

# Cascade delete - get all IDs for cleanup
ids = table.unlink_and_get_ids("rescue_event_001")
# -> Delete from ChromaDB, GraphStore, DuckDB using returned IDs
```

### Graph Store - Entity Relationships

```python
from components.stores.duckdb_store import GraphStore

graph = GraphStore(config={"path": "graph.duckdb"})

# Add entities and relationships
graph.add_node("person:alice", "person", {"name": "Alice"})
graph.add_node("person:bob", "person", {"name": "Bob"})
graph.add_edge("person:alice", "knows", "person:bob")

# Find relationships
neighbors = graph.find_neighbors("person:alice", direction="outgoing")
path = graph.find_path("person:alice", "person:charlie")
```

### Working Memory - TTL Buffer

```python
from components.stores.duckdb_store import WorkingMemory

memory = WorkingMemory(config={
    "path": "working.duckdb",
    "ttl_seconds": 3600,  # 1 hour
    "max_size": 10000,
})

# Add streaming data
memory.add("chat", "User asked about weather", {"user_id": "123"})
memory.add("telemetry", "Heart rate: 72", {"device": "watch"})

# Get recent context
recent = memory.get_recent(limit=10, minutes=5)
chats = memory.get_by_type("chat", limit=20)

# Expired records are auto-pruned
```

## Architecture

```
                    ┌────────────────────────────────────┐
                    │         Unified MemoryStore        │
                    └─────────────┬──────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  ChromaDB     │       │    DuckDB       │       │    DuckDB       │
│  (Vector)     │       │  (TimeSeries)   │       │   (Graph)       │
└───────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────────────┐
                    │         Linkage Table              │
                    │  UUID → {vector, graph, ts IDs}    │
                    └────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- DuckDB (installed via `uv add duckdb`)
- ChromaDB (for vector operations)
