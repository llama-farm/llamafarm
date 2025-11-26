---
title: RAG Guide
sidebar_position: 6
---

# RAG Guide

LlamaFarm treats retrieval-augmented generation as a first-class, configurable pipeline. This guide explains how strategies, databases, and datasets fit together—and how to operate and extend them.

## RAG at a Glance

| Piece | Where it lives | Purpose |
| ----- | -------------- | ------- |
| `rag.databases[]` | `llamafarm.yaml` | Define vector stores and retrieval strategies. |
| `rag.data_processing_strategies[]` | `llamafarm.yaml` | Describe parsers, extractors, and metadata processors for ingestion. |
| `lf datasets create/upload/process` | CLI | Ingest documents according to the chosen strategy/database. |
| `lf rag query` | CLI | Query the store with semantic, hybrid, or metadata-aware retrieval. |
| Celery workers | Server runtime | Perform heavy ingestion tasks. |

## Configure Databases

Each database entry declares a store type (default `ChromaStore` or `QdrantStore`) and the embedding/retrieval strategies available.

```yaml
rag:
  databases:
    - name: main_db
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: semantic_search
      embedding_strategies:
        - name: default_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text:latest
      retrieval_strategies:
        - name: semantic_search
          type: VectorRetriever
          config:
            top_k: 5
        - name: hybrid_search
          type: HybridUniversalStrategy
          config:
            dense_weight: 0.6
            sparse_weight: 0.4
```

- Add multiple strategies for different workloads (semantic, keyword, reranked).
- Set `default_*` fields to control CLI defaults.
- Extend store/types by editing `rag/schema.yaml` and following the [Extending guide](../extending/index.md#extend-rag-components).

## Define Processing Strategies

Processing strategies control how files become chunks in the vector store.

```yaml
rag:
  data_processing_strategies:
    - name: pdf_ingest
      description: Ingest FDA letters with headings & stats.
      parsers:
        - type: PDFParser_LlamaIndex
          config:
            chunk_size: 1500
            chunk_overlap: 200
            preserve_layout: true
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor
      metadata_extractors:
        - type: EntityExtractor
```

- Parsers handle format-aware chunking (PDF, CSV, DOCX, Markdown, text).
- Extractors add metadata (entities, headings, statistics) to each chunk.
- Customize chunk size/overlap per parser type.

## Dataset Lifecycle

1. **Create** a dataset referencing a strategy and database.
   ```bash
   lf datasets create -s pdf_ingest -b main_db research-notes
   ```
2. **Upload** files via `lf datasets upload` (supports globs and directories). The CLI stores file hashes for dedupe.
3. **Process** documents with `lf datasets process research-notes`. The server schedules a Celery job; monitor progress in the CLI output and server logs.
4. **Query** with `lf rag query` to validate retrieval quality.

## Querying & Retrieval Strategies

`lf rag query` exposes several toggles:

- `--retrieval-strategy` to select among those defined in the database.
- `--filter "key:value"` for metadata filtering (e.g., `doc_type:letter`, `year:2024`).
- `--top-k`, `--score-threshold`, `--include-metadata`, `--include-score` for result tuning.
- `--distance-metric`, `--hybrid-alpha`, `--rerank-model`, `--query-expansion` for advanced workflows.

Pair queries with `lf chat` to confirm the runtime consumes retrieved context correctly.

### Advanced Retrieval Strategies

For improved accuracy and handling of complex queries, LlamaFarm supports:

- **Cross-Encoder Reranking** - Rerank initial candidates using specialized models (10-100x faster than LLM reranking)
- **Multi-Turn RAG** - Decompose complex queries into sub-queries, retrieve in parallel, and merge results

See [Advanced Retrieval Strategies](./advanced-retrieval.md) for detailed configuration and usage.

## Monitoring & Maintenance

- `lf rag stats` – view vector counts and storage usage.
- `lf rag health` – check embedder/store health status.
- `lf rag list` – inspect documents and metadata.
- `lf rag compact` / `lf rag reindex` – maintain store performance.
- `lf rag clear` / `lf rag delete` – remove data (dangerous; confirm before use).

## Troubleshooting

| Symptom | Possible Cause | Fix |
| ------- | -------------- | ---- |
| `No response received` after `lf chat` | Runtime returned empty stream (model mismatch, tool support) | Try `--no-rag`, switch models, or adjust agent handler. |
| `Task timed out or failed: PENDING` during processing | Celery worker still ingesting large files | Wait and re-run, check worker logs, ensure enough resources. |
| Query returns 0 results | Incorrect strategy/database, unprocessed dataset, high score threshold | Verify dataset processed successfully, adjust `--score-threshold`. |

## Next Steps

- [CLI Reference](../cli/index.md) – command usage.
- [Extending RAG](../extending/index.md#extend-rag-components) – add stores/parsers.
- [Examples](../examples/index.md) – see FDA and Raleigh workflows.
