# Mixed-Format RAG Example

Simulates analysing a project knowledge base that mixes PDFs, Markdown guides, HTML articles, raw text, and code snippets.

## What This Demo Shows
- How the ingestion pipeline handles heterogeneous file types.
- Hybrid retrieval for combining dense and sparse signals.
- Citation-heavy responses that reference the underlying file names.
- Contrast between RAG-augmented answers and baseline LLM output.

## Contents
- `files/` – sample documents (FDA letter, research note, Markdown API guide, HTML article, Python source).
- `llamafarm.yaml` – example configuration with mixed parsers/extractors, consumed automatically via `lf --cwd`.
- `run_example.sh` – interactive workflow using the latest CLI commands.

## Quickstart
```bash
# From the repo root (expects ./lf or export LF_BIN=/full/path/to/lf)
./examples/mixed-format-rag/run_example.sh

# Optional: pass the directory containing the lf binary if you keep it elsewhere
./examples/mixed-format-rag/run_example.sh /path/to/your/project
```

Set `NO_PAUSE=1` to skip prompts (useful for CI or batch runs).

## Manual Workflow
```bash
# Create dataset pointing at the mixed-format processing strategy and new database
lf --cwd examples/mixed-format-rag datasets create -s mixed_content_processor -b mixed_format_db mixed_format_dataset

# Upload documents (PDF, Markdown, HTML, text, code)
lf --cwd examples/mixed-format-rag datasets upload mixed_format_dataset examples/mixed-format-rag/files/*

# Process and inspect
lf --cwd examples/mixed-format-rag datasets process mixed_format_dataset
lf --cwd examples/mixed-format-rag datasets list
lf --cwd examples/mixed-format-rag rag query --database mixed_format_db --top-k 4 --include-metadata --include-score \
  "Summarize transformer architecture mentions across documents."

# Ask questions with RAG context
lf --cwd examples/mixed-format-rag chat --database mixed_format_db "Provide an overview of transformer architecture with citations."
lf --cwd examples/mixed-format-rag chat --database mixed_format_db "List documented API endpoints with file references."

# Compare baseline
lf --cwd examples/mixed-format-rag chat --no-rag "What is transformer architecture?"
```

## Cleanup
```bash
lf --cwd examples/mixed-format-rag datasets delete mixed_format_dataset
rm -rf examples/mixed-format-rag/data/mixed_format_db
```
