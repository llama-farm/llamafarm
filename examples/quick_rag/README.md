# Quick RAG Example

Two small engineering notes illustrate how fast it is to ingest and query short documents.

## Highlights
- Minimal ingestion time (two files totalling a few kilobytes).
- Great for sanity-checking your environment or demoing the CLI.
- Demonstrates contrast between RAG-augmented answers and baseline LLM output.

## Contents
- `files/` – Markdown + text notes about neural scaling laws and engineering best practices.
- `llamafarm.yaml` – lightweight config automatically used when the CLI runs with `--cwd` pointing to this directory.
- `run_example.sh` – interactive walkthrough using the latest CLI commands.

## Quickstart
```bash
# From the repo root (expects ./lf or export LF_BIN=/full/path/to/lf)
./examples/quick_rag/run_example.sh

# Optional: target a different repo/build folder that holds the lf binary
./examples/quick_rag/run_example.sh /path/to/your/project
```
Set `NO_PAUSE=1` to skip the “press Enter” prompts.

## Manual Workflow
```bash
lf --cwd examples/quick_rag datasets create -s quick_note_processor -b quick_rag_db quick_dataset
lf --cwd examples/quick_rag datasets upload quick_dataset examples/quick_rag/files/*
lf --cwd examples/quick_rag datasets process quick_dataset

# Retrieval & chat
lf --cwd examples/quick_rag rag query --database quick_rag_db --top-k 3 --include-metadata --include-score \
  "Reference material that discusses neural scaling laws."
lf --cwd examples/quick_rag chat --database quick_rag_db "Summarize neural scaling laws in two sentences with citations."
lf --cwd examples/quick_rag chat --no-rag "Summarize neural scaling laws in two sentences."
```

## Cleanup
```bash
lf --cwd examples/quick_rag datasets delete quick_dataset
rm -rf examples/quick_rag/data/quick_rag_db
```
