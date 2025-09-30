# LlamaFarm Examples

Four ready-to-run demos showcase different retrieval scenarios using the latest `lf` CLI.

| Example | Folder | Scenario |
|---------|--------|----------|
| Large, Complex PDFs | `examples/large-complex-rag/` | Raleigh UDO ordinance (multi-megabyte planning PDF). |
| Many Small Files | `examples/many-small-file-rag/` | FDA correspondence letters (several shorter PDFs). |
| Mixed Formats | `examples/mixed-format-rag/` | Blend of PDF, Markdown, HTML, text, and code. |
| Quick Notes | `examples/quick-rag/` | Two tiny engineering notes for rapid smoke tests. |

Each directory contains:
- `files/` – sample documents.
- `llamafarm.yaml` – scenario-specific configuration automatically loaded when you run `lf --cwd <example_dir>`.
- `run_example.sh` – interactive script (press Enter between steps) that uses `lf datasets`, `lf rag`, and `lf chat` while scoping the CLI to the example directory.

> Set `NO_PAUSE=1` when running scripts if you prefer non-interactive output (CI, automation, etc.).

## Prerequisites
- CLI installed or built (`curl … install.sh` or `go build -o lf ./cli`).
- Docker + Ollama running (or manual services via `nx dev`).
- Embedding model available in Ollama (e.g., `ollama pull nomic-embed-text`).

## Manual Workflow Cheat Sheet
```bash
# Create dataset (strategy + database)
lf --cwd examples/<folder> datasets create -s <strategy> -b <database> <dataset>

# Upload files
lf --cwd examples/<folder> datasets upload <dataset> examples/<folder>/files/*

# Process and inspect
lf --cwd examples/<folder> datasets process <dataset>
lf --cwd examples/<folder> datasets list
lf --cwd examples/<folder> rag query --database <database> --top-k 3 --include-metadata --include-score "Your question"

# Ask questions
lf --cwd examples/<folder> chat --database <database> "Prompt with citations"
lf --cwd examples/<folder> chat --no-rag "Same prompt without RAG"

# Cleanup
lf --cwd examples/<folder> datasets delete <dataset>
rm -rf examples/<folder>/data/<database>
```

Refer to each example’s README for scenario-specific prompts and cleanup guidance.
