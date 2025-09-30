# Mixed-Format RAG Example

Simulates analysing a project knowledge base that mixes PDFs, Markdown guides, HTML articles, raw text, and code snippets.

## What This Demo Shows
- How the ingestion pipeline handles heterogeneous file types.
- Hybrid retrieval for combining dense and sparse signals.
- Citation-heavy responses that reference the underlying file names.
- Contrast between RAG-augmented answers and baseline LLM output.

## Contents
- `files/` – sample documents (FDA letter, research note, Markdown API guide, HTML article, Python source).
- `llamafarm-example-mixed-format.yaml` – example configuration with mixed parsers/extractors.
- `update_config.sh` – copies the config into your project (backs up existing config).
- `run_example.sh` – interactive workflow using the latest CLI commands.

## Quickstart
```bash
# Optional: initialize or select a project directory
mkdir -p ~/projects/mixed-demo && cd ~/projects/mixed-demo
lf init

# Apply the example configuration (backs up existing llamafarm.yaml)
/path/to/llamafarm/examples/mixed-format-rag/update_config.sh /path/to/your/project

# Run the interactive workflow (press Enter between steps)
/path/to/llamafarm/examples/mixed-format-rag/run_example.sh /path/to/your/project
```

Set `NO_PAUSE=1` to skip prompts (useful for CI or batch runs).

## Manual Workflow
```bash
# Create dataset pointing at the mixed-format processing strategy and new database
lf datasets create -s mixed_content_processor -b mixed_format_db mixed_format_dataset

# Upload documents (PDF, Markdown, HTML, text, code)
lf datasets upload mixed_format_dataset examples/mixed-format-rag/files/*

# Process and inspect
lf datasets process mixed_format_dataset
lf datasets list
lf rag query --database mixed_format_db --top-k 4 --include-metadata --include-score \
  "Summarize transformer architecture mentions across documents."

# Ask questions with RAG context
lf chat --database mixed_format_db "Provide an overview of transformer architecture with citations."
lf chat --database mixed_format_db "List documented API endpoints with file references."

# Compare baseline
lf chat --no-rag "What is transformer architecture?"
```

## Cleanup
```bash
lf datasets delete mixed_format_dataset
rm -rf data/mixed_format_db
```

Remember to restore your config from the backup created by `update_config.sh` if you no longer want the example settings.
