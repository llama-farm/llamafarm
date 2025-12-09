---
title: Databases Reference
sidebar_position: 5
---

# Databases (Vector Stores) Reference

Vector databases store embeddings and enable semantic search. LlamaFarm supports multiple vector store backends for different deployment scenarios.

## Quick Start

Databases are configured in `rag.databases`:

```yaml
rag:
  default_database: main_db
  databases:
    - name: main_db
      type: ChromaStore
      config:
        collection_name: documents
        distance_function: cosine
```

### Common Database Properties

| Property | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Unique identifier (lowercase, underscores) |
| `type` | Yes | Store type (`ChromaStore`, `QdrantStore`, etc.) |
| `config` | No | Store-specific configuration |
| `embedding_strategies` | No | List of embedding configurations |
| `retrieval_strategies` | No | List of retrieval configurations |
| `default_embedding_strategy` | No | Default embedder name |
| `default_retrieval_strategy` | No | Default retrieval strategy name |

---

## ChromaStore

Chroma is a lightweight, embedded vector database. Great for development and small-to-medium deployments.

**Best for:** Local development, prototyping, embedded use

```yaml
- name: main_db
  type: ChromaStore
  config:
    collection_name: documents
    distance_function: cosine
    persist_directory: ./data/chroma_db
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `collection_name` | string | `documents` | Collection name (alphanumeric, hyphens, underscores) |
| `host` | string | `null` | Server host (null for embedded) |
| `port` | integer | 8000 | Server port |
| `distance_function` | string | `cosine` | `cosine`, `l2`, `ip` |
| `distance_metric` | string | `cosine` | Alternative name for distance_function |
| `embedding_dimension` | integer | 768 | Vector dimension (1-4096) |
| `enable_deduplication` | boolean | `true` | Enable document deduplication |
| `embedding_function` | string | `null` | Built-in embedding function |

### Distance Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `cosine` | Cosine similarity | General purpose, normalized vectors |
| `l2` | Euclidean distance | When magnitude matters |
| `ip` | Inner product | Already normalized embeddings |

### Client vs Server Mode

**Embedded (default):**
```yaml
config:
  collection_name: documents
  persist_directory: ./data/chroma
```

**Server mode:**
```yaml
config:
  collection_name: documents
  host: localhost
  port: 8000
```

---

## QdrantStore (Coming Soon)

:::note Coming Soon
QdrantStore is planned but not yet implemented. Use **ChromaStore** for now.
:::

Qdrant is a high-performance vector database with rich filtering and clustering.

**Best for:** Production deployments, large datasets, advanced filtering

```yaml
- name: production_db
  type: QdrantStore
  config:
    collection_name: documents
    host: localhost
    port: 6333
    vector_size: 768
    distance: Cosine
```

### Planned Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `collection_name` | string | `documents` | No | Collection name |
| `host` | string | `localhost` | No | Server host |
| `port` | integer | 6333 | No | REST API port |
| `grpc_port` | integer | 6334 | No | gRPC port |
| `api_key` | string | `null` | No | API key for auth |
| `vector_size` | integer | - | Yes | Vector dimension (1-65536) |
| `distance` | string | `Cosine` | No | `Cosine`, `Euclid`, `Dot` |
| `on_disk` | boolean | `false` | No | Store vectors on disk |

### Distance Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `Cosine` | Cosine similarity | General semantic search |
| `Euclid` | Euclidean distance | Geometric comparisons |
| `Dot` | Dot product | Pre-normalized vectors |

### Docker Setup (for when available)

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

---

## FAISSStore (Coming Soon)

:::note Coming Soon
FAISSStore is planned but not yet implemented. Use **ChromaStore** for now.
:::

FAISS (Facebook AI Similarity Search) for high-performance similarity search.

**Best for:** Large-scale search, research, GPU acceleration

```yaml
- name: faiss_db
  type: FAISSStore
  config:
    dimension: 768
    index_type: HNSW
    metric: Cosine
```

### Planned Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `dimension` | integer | - | Yes | Vector dimension (1-4096) |
| `index_type` | string | `Flat` | No | `Flat`, `IVF`, `HNSW`, `LSH` |
| `metric` | string | `L2` | No | `L2`, `IP`, `Cosine` |
| `nlist` | integer | 100 | No | Number of clusters (IVF) |
| `nprobe` | integer | 10 | No | Clusters to search (IVF) |
| `use_gpu` | boolean | `false` | No | Enable GPU acceleration |

### Index Types

| Type | Speed | Accuracy | Memory | Best For |
|------|-------|----------|--------|----------|
| `Flat` | Slow | Exact | High | Small datasets, exact search |
| `IVF` | Fast | Good | Medium | Large datasets |
| `HNSW` | Very Fast | Very Good | High | Production, low latency |
| `LSH` | Fast | Moderate | Low | Very large datasets |

---

## PineconeStore (Coming Soon)

:::note Coming Soon
PineconeStore is planned but not yet implemented. Use **ChromaStore** for now.
:::

Pinecone is a fully managed cloud vector database.

**Best for:** Managed service, enterprise, global scale

```yaml
- name: cloud_db
  type: PineconeStore
  config:
    api_key: ${PINECONE_API_KEY}
    index_name: my-index
    dimension: 768
    environment: us-east-1-aws
```

### Planned Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `api_key` | string | - | Yes | Pinecone API key |
| `index_name` | string | - | Yes | Index name (lowercase, hyphens) |
| `dimension` | integer | - | Yes | Vector dimension (1-20000) |
| `environment` | string | `us-east-1-aws` | No | Pinecone environment |
| `metric` | string | `cosine` | No | `euclidean`, `cosine`, `dotproduct` |
| `namespace` | string | `""` | No | Namespace for isolation |
| `replicas` | integer | 1 | No | Number of replicas (1-20) |

---

## Complete Database Configuration

Full example with embedding and retrieval strategies:

```yaml
rag:
  default_database: main_db

  databases:
    - name: main_db
      type: ChromaStore
      config:
        collection_name: documents
        distance_function: cosine
        persist_directory: ./data/main_db

      # Embedding strategies
      embedding_strategies:
        - name: default
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
            dimension: 768
          priority: 0

        - name: openai
          type: OpenAIEmbedder
          config:
            model: text-embedding-3-small
            api_key: ${OPENAI_API_KEY}
          priority: 1

      # Retrieval strategies
      retrieval_strategies:
        - name: semantic
          type: BasicSimilarityStrategy
          config:
            distance_metric: cosine
            top_k: 10
          default: true

        - name: filtered
          type: MetadataFilteredStrategy
          config:
            top_k: 10
            filter_mode: pre

      default_embedding_strategy: default
      default_retrieval_strategy: semantic
```

## Multiple Databases

Configure separate databases for different use cases:

```yaml
databases:
  # Primary document store
  - name: documents_db
    type: ChromaStore
    config:
      collection_name: documents

  # Archive for older content (embedded mode)
  - name: archive_db
    type: ChromaStore
    config:
      collection_name: archive
      persist_directory: ./data/archive

  # Server-mode for production
  - name: production_db
    type: ChromaStore
    config:
      collection_name: production
      host: localhost
      port: 8000
```

:::tip Future Options
When QdrantStore and FAISSStore are available, you'll be able to use them for larger scale deployments and more advanced features.
:::

Query specific databases:

```bash
lf rag query --database documents_db "recent reports"
lf rag query --database archive_db "historical data"
```

## Database Selection Guide

| Database | Deployment | Scale | Features | Status |
|----------|------------|-------|----------|--------|
| ChromaStore | Embedded/Server | Small-Medium | Simple, portable | **Available** |
| QdrantStore | Self-hosted | Medium-Large | Rich filtering, clustering | Coming Soon |
| FAISSStore | Self-hosted | Very Large | GPU support, research | Coming Soon |
| PineconeStore | Cloud | Any | Managed, global | Coming Soon |

## Next Steps

- [Retrieval Strategies](./retrieval-strategies.md) - Configure retrieval methods
- [Embedders Reference](./embedders.md) - Configure embeddings
- [RAG Guide](./index.md) - Full RAG overview
