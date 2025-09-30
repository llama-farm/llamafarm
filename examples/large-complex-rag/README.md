# Large, Complex RAG Example (FDA Correspondence)

Process multi-megabyte FDA correspondence letters and explore how RAG handles dense regulatory content.

## What This Demo Highlights
- Parsing and chunking long PDFs with multiple extractors.
- Creating isolated databases/datasets per run to avoid stale state.
- Example questions that require citations and cross-letter comparisons.
- Contrast between RAG-augmented responses and baseline LLM output.

## Directory Contents
- `files/` – three FDA Complete Response letters (PDFs).
- `llamafarm-example-large-complex.yaml` – configuration tuned for large PDFs.
- `update_config.sh` – copies the config into your project (backs up existing config).
- `run_example.sh` – interactive workflow using the latest CLI commands (`lf chat`, `lf datasets`, etc.).

## Quickstart
```bash
# Optional: initialize a project directory
lf init fda-demo

# Copy the example configuration
/path/to/llamafarm/examples/large-complex-rag/update_config.sh /path/to/your/project

# Run the interactive script (press Enter between steps)
/path/to/llamafarm/examples/large-complex-rag/run_example.sh /path/to/your/project
```
Set `NO_PAUSE=1` to skip prompts during automated runs.

The script clones a base database strategy, creates unique database/dataset names, uploads PDFs, processes them, runs retrieval, and issues several `lf chat` prompts with and without RAG. Each command’s output is printed so you can inspect chunk counts, metadata, and responses.

## Manual Workflow
```bash
# Create a dataset and point to the large PDF processor
lf datasets create -s fda_pdf_processor -b fda_letters_db fda_letters

# Upload the letters
lf datasets upload fda_letters examples/large-complex-rag/files/*.pdf

# Process and inspect
lf datasets process fda_letters
lf datasets list
lf rag query --database fda_letters_db --top-k 3 --include-metadata --include-score \
  "Which FDA letters mention additional clinical trial data requirements?"

# Ask questions with RAG
lf chat --database fda_letters_db "Summarize key deficiencies highlighted in the 2024 letters with citations."
lf chat --database fda_letters_db "According to correspondence 761240, what follow-up actions were requested?"

# Baseline comparison
lf chat --no-rag "According to correspondence 761240, what follow-up actions were requested?"
```

## Cleanup
```bash
lf datasets delete fda_letters
rm -rf data/fda_letters_db
```
Restore the configuration backup created by `update_config.sh` if you no longer need the example settings.
