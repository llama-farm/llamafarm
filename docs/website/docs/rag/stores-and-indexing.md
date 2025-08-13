---
title: Stores & indexing
sidebar_label: Stores & indexing
slug: /rag/stores-and-indexing
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Choosing a vector store and building efficient indexes.

## Stores

Chroma, FAISS, others.

## Index build

Chunking, embeddings, incremental updates.

## Maintenance

Compaction, re-embed strategies, drift.

### Flexible path resolution (tips)

```bash
# Global options for CLI
--config, -c     Configuration file path (default: rag_config.json)
--base-dir, -b   Base directory for relative path resolution
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR)
```

```bash
# Relative paths (resolved from current or base directory)
uv run python cli.py ingest samples/small_sample.csv
uv run python cli.py ingest data/tickets.csv

# Absolute paths
uv run python cli.py ingest /path/to/data/tickets.csv

# Home directory expansion
uv run python cli.py ingest ~/Documents/support_data.csv

# With custom base directory
uv run python cli.py --base-dir /project/root ingest data/tickets.csv
```

```bash
# Set custom base directory for all relative paths
uv run python cli.py --base-dir /my/project/root \
  --config configs/prod.json \
  ingest data/latest_tickets.csv
```
