---
title: Examples
sidebar_position: 10
---

# Example Workflows

The repository ships with interactive demos that highlight different retrieval scenarios. Each example lives under `examples/<folder>` and provides a configuration, sample data, and a script that uses the newest CLI commands (e.g., `lf datasets create`, `lf chat`).

| Folder | Use Case | Highlights |
|--------|----------|------------|
| `large-complex-rag/` | Multi-megabyte FDA correspondence PDFs | Long-running ingestion, citation-heavy prompts, unique DB/dataset per run. |
| `many-small-file-rag/` | Raleigh UDO ordinance | Large single PDF, long processing time, zoning queries. |
| `mixed-format-rag/` | Blend of PDF/Markdown/HTML/text/code | Hybrid retrieval, multiple parsers/extractors in one pipeline. |
| `quick-rag/` | Two short engineering notes | Rapid smoke test for the environment and CLI. |

## How to Run an Example
```bash
# Optional: initialize a project
gor init my-project
lf init

# Apply an example configuration (backs up existing llamafarm.yaml)
./examples/<folder>/update_config.sh /path/to/your/project

# Run the interactive workflow (press Enter between steps)
./examples/<folder>/run_example.sh /path/to/your/project

# Skip prompts if desired
NO_PAUSE=1 ./examples/<folder>/run_example.sh
```

Each script clones the relevant database entry, creates a unique dataset/database pair, uploads the sample documents, processes them, prints the CLI output verbatim, runs meaningful `lf rag query` and `lf chat` commands, and finishes with a baseline `--no-rag` comparison. Clean-up instructions are printed at the end of each script.

## Manual Command Reference
Use these commands if you prefer to run the workflows yourself:
```bash
lf datasets create -s <strategy> -b <database> <dataset>
lf datasets upload <dataset> path/to/files/*
lf datasets process <dataset>
lf rag query --database <database> --top-k 3 --include-metadata --include-score "Your question"
lf chat --database <database> "Prompt needing citations"
lf chat --no-rag "Same prompt without RAG"
lf datasets delete <dataset>
rm -rf data/<database>
```

Refer to each example folder’s README for scenario-specific prompts, cleanup suggestions, and contextual background (e.g., why those documents were chosen and what use cases they simulate).
