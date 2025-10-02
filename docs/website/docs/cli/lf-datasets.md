---
title: lf datasets
sidebar_position: 4
---

# `lf datasets`

Manage datasets defined for your project—create, upload, process, list, and delete ingestion targets used by RAG.

## Subcommands

| Command | Description |
| ------- | ----------- |
| `lf datasets list` | Show datasets for the current project. |
| `lf datasets create` | Create a dataset (optionally upload files immediately). |
| `lf datasets upload` | Upload files to an existing dataset. |
| `lf datasets process` | Run the ingestion pipeline (parsers, extractors, embeddings). |
| `lf datasets delete` | Remove a dataset. |

Each subcommand accepts the global flags for server resolution and config watching.

## `lf datasets list`

```
lf datasets list
```

Prints a table containing dataset name, strategy, database, and file count.

## `lf datasets create`

```
lf datasets create -s <strategy> -b <database> <name> [files...]
```

- Validates that the strategy and database exist in your `rag` configuration.
- Accepts globs and directories for optional initial upload (same semantics as `upload`).

## `lf datasets upload`

```
lf datasets upload <name> <path> [paths...]
```

- Supports directories, globs, and individual files.
- Displays per-file success/failure alongside a summary.

## `lf datasets process`

```
lf datasets process <name>
```

- Sends an ingestion job to Celery.
- Shows progress dots every 2 seconds when stdout is a TTY.
- Reads results and prints chunk counts, parser/extractor usage, and skip reasons (duplicates, errors).

If you see a timeout, the worker may still be processing—wait and re-run. Consult server logs for detailed status.

## `lf datasets delete`

```
lf datasets delete <name>
```

Deletes the dataset and its metadata from the server (stored files remain unless back-end removes them).

## Tips

- Place shared data under `datasets` in `llamafarm.yaml` if you want the CLI to sync metadata.
- Run `lf rag health` after processing to ensure embedder/store are healthy.
- Combine with `lf rag query` to verify retrieval quality immediately after ingestion.

## See Also

- [RAG Guide](../rag/index.md)
- [Troubleshooting](../troubleshooting/index.md)
