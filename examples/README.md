# LlamaFarm Examples

Four ready-to-run demos showcase different retrieval scenarios using the latest `lf` CLI.

| Example | Folder | Scenario |
|---------|--------|----------|
| Large, Complex PDFs | `examples/large-complex-rag/` | FDA correspondence letters (multi-megabyte PDFs). |
| Many Small Files | `examples/many-small-file-rag/` | Raleigh UDO ordinance (long-form regulatory PDF). |
| Mixed Formats | `examples/mixed-format-rag/` | Blend of PDF, Markdown, HTML, text, and code. |
| Quick Notes | `examples/quick-rag/` | Two tiny engineering notes for rapid smoke tests. |

Each directory contains:
- `files/` – sample documents.
- `llamafarm-example-*.yaml` – configuration tuned for the scenario.
- `update_config.sh` – copies the config into your project (backs up existing config).
- `run_example.sh` – interactive script (press Enter between steps) that uses `lf datasets`, `lf rag`, and `lf chat`.

> Set `NO_PAUSE=1` when running scripts if you prefer non-interactive output (CI, automation, etc.).

## Prerequisites
- CLI installed or built (`curl … install.sh` or `go build -o lf ./cli`).
- Docker + Ollama running (or manual services via `nx dev`).
- Embedding model available in Ollama (e.g., `ollama pull nomic-embed-text`).

## Manual Workflow Cheat Sheet
```bash
# Create dataset (strategy + database)
lf datasets create -s <strategy> -b <database> <dataset>

# Upload files
lf datasets upload <dataset> path/to/documents/*

# Process and inspect
lf datasets process <dataset>
lf datasets list
lf rag query --database <database> --top-k 3 --include-metadata --include-score "Your question"

# Ask questions
lf chat --database <database> "Prompt with citations"
lf chat --no-rag "Same prompt without RAG"

# Cleanup
lf datasets delete <dataset>
rm -rf data/<database>
```

Refer to each example’s README for scenario-specific prompts and cleanup guidance.
