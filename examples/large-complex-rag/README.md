# Large, Complex RAG Example (Raleigh UDO)

Process the Raleigh Unified Development Ordinance (UDO) — a long, highly structured municipal document — and explore how RAG handles dense regulatory content.

## What This Demo Highlights
- Parsing and chunking long PDFs with multiple extractors.
- Creating isolated databases/datasets per run to avoid stale state.
- Example questions that require citations and cross-letter comparisons.
- Contrast between RAG-augmented responses and baseline LLM output.

## Directory Contents
- `files/` – the Raleigh UDO supplemental PDF used in this demo.
- `llamafarm.yaml` – configuration tuned for large ordinance PDFs, consumed automatically via `lf --cwd`.
- `run_example.sh` – interactive workflow using the latest CLI commands (`lf chat`, `lf datasets`, etc.).

## Quickstart
```bash
# From the repo root (expects ./lf or export LF_BIN=/full/path/to/lf)
./examples/large-complex-rag/run_example.sh

# Optional: run against a different checkout/build directory that contains the lf binary
./examples/large-complex-rag/run_example.sh /path/to/your/project
```
Set `NO_PAUSE=1` to skip prompts during automated runs.

The script clones a base database strategy, creates unique database/dataset names, uploads PDFs, processes them, runs retrieval, and issues several `lf chat` prompts with and without RAG. Each command’s output is printed so you can inspect chunk counts, metadata, and responses.

## Manual Workflow
```bash
# Create a dataset and point to the UDO PDF processor
lf --cwd examples/large-complex-rag datasets create -s udo_pdf_processor -b raleigh_udo_db raleigh_udo_dataset

# Upload the ordinance PDF
lf --cwd examples/large-complex-rag datasets upload raleigh_udo_dataset examples/large-complex-rag/files/*.pdf

# Process and inspect
lf --cwd examples/large-complex-rag datasets process raleigh_udo_dataset
lf --cwd examples/large-complex-rag datasets list
lf --cwd examples/large-complex-rag rag query --database raleigh_udo_db --top-k 3 --include-metadata --include-score \
  "Which section of the ordinance covers neighborhood transition requirements?"

# Ask questions with RAG
lf --cwd examples/large-complex-rag chat --database raleigh_udo_db "Summarize height allowances for Village Mixed Use districts with citations."
lf --cwd examples/large-complex-rag chat --database raleigh_udo_db "List buffering requirements when non-residential development abuts residential lots."

# Baseline comparison
lf --cwd examples/large-complex-rag chat --no-rag "How tall can buildings be in Village Mixed Use zoning?"
```

## Cleanup
```bash
lf --cwd examples/large-complex-rag datasets delete raleigh_udo_dataset
rm -rf examples/large-complex-rag/data/raleigh_udo_db
```
