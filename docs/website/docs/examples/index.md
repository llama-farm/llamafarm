---
title: Examples
sidebar_position: 10
---

# Example Workflows

The repository ships with interactive demos that highlight different retrieval scenarios. Each example lives under `examples/<folder>` and provides a configuration, sample data, and a script that uses the newest CLI commands (e.g., `lf datasets create`, `lf chat`).

| Folder | Use Case | Highlights |
|--------|----------|------------|
| `large-complex-rag/` | Multi-megabyte Raleigh UDO ordinance PDF | Long-running ingestion, zoning-focused prompts, unique DB/dataset per run. |
| `many-small-file-rag/` | FDA correspondence packet | Several shorter PDFs, quick iteration, letter-specific queries. |
| `mixed-format-rag/` | Blend of PDF/Markdown/HTML/text/code | Hybrid retrieval, multiple parsers/extractors in one pipeline. |
| `quick-rag/` | Two short engineering notes | Rapid smoke test for the environment and CLI. |

## How to Run an Example
```bash
# Build or install the CLI if needed
go build -o lf ./cli

# Run the interactive workflow (press Enter between steps).
# The script automatically scopes the CLI with `lf --cwd examples/<folder>`.
./examples/<folder>/run_example.sh

# Optional: point the script at a different directory that contains the lf binary
./examples/<folder>/run_example.sh /path/to/your/project

# Skip prompts if desired
NO_PAUSE=1 ./examples/<folder>/run_example.sh
```

Each script clones the relevant database entry, creates a unique dataset/database pair, uploads the sample documents, processes them, prints the CLI output verbatim, runs meaningful `lf rag query` and `lf chat` commands, and finishes with a baseline `--no-rag` comparison. Clean-up instructions are printed at the end of each script.

## Manual Command Reference
Use these commands if you prefer to run the workflows yourself (replace `<folder>` with the example you want to explore):
```bash
lf --cwd examples/<folder> datasets create -s <strategy> -b <database> <dataset>
lf --cwd examples/<folder> datasets upload <dataset> examples/<folder>/files/*
lf --cwd examples/<folder> datasets process <dataset>
lf --cwd examples/<folder> rag query --database <database> --top-k 3 --include-metadata --include-score "Your question"
lf --cwd examples/<folder> chat --database <database> "Prompt needing citations"
lf --cwd examples/<folder> chat --no-rag "Same prompt without RAG"
lf --cwd examples/<folder> datasets delete <dataset>
rm -rf examples/<folder>/data/<database>
```

Refer to each example folder’s README for scenario-specific prompts, cleanup suggestions, and contextual background (e.g., why those documents were chosen and what use cases they simulate).
