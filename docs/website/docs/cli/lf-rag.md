---
title: lf rag
sidebar_position: 5
---

# `lf rag`

Query your knowledge base and access RAG maintenance utilities.

## Querying Documents

```
lf rag query "question" [flags]
```

| Flag | Purpose |
| ---- | ------- |
| `--database` | Select a database (defaults to config default). |
| `--data-processing-strategy` | Filter results to a strategy. |
| `--retrieval-strategy` | Override retrieval behaviour (vector, hybrid, metadata filtered, etc.). |
| `--top-k` | Number of chunks to return. |
| `--score-threshold` | Minimum similarity score. |
| `--filter` | Apply metadata filters (`key:value`). Repeatable. |
| `--include-metadata`, `--include-score` | Show metadata/score columns. |
| `--distance-metric`, `--hybrid-alpha`, `--rerank-model`, `--query-expansion`, `--max-tokens` | Advanced knobs matching server capabilities. |

Example:

```bash
lf rag query --database main_db --filter "doc_type:letter" --include-metadata \
  "Which letters mention additional clinical trials?"
```

## Maintenance Commands

Some subcommands are hidden from `--help` but available for operators:

| Command | Description |
| ------- | ----------- |
| `lf rag stats` | Vector/document counts, storage usage (JSON or table). |
| `lf rag health` | Embedder/store health summary. |
| `lf rag list` | List ingested documents and metadata. |
| `lf rag compact` | Compact/optimize the vector store. |
| `lf rag reindex` | Reindex all documents using a given strategy. |
| `lf rag clear` | Delete **all** documents from a database (dangerous). |
| `lf rag delete` | Remove documents by ID, filename, or metadata filter. |
| `lf rag export/import` | Move datasets between environments. |

> ⚠️ Destructive commands (`clear`, `delete`) prompt for confirmation unless you pass `--force`.

## Troubleshooting

- **Empty results** – confirm dataset processing succeeded and the retrieval strategy matches your query type.
- **Timeouts** – large datasets can take time to process; check Celery logs or increase `--server-start-timeout` before retrying.
- **Hybrid/Tool Errors** – some smaller models don’t support tool calls; switch to a basic agent handler via configuration.

## See Also

- [RAG Guide](../rag/index.md)
- [Extending RAG](../extending/index.md#extend-rag-components)
