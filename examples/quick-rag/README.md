# Quick RAG Example

Two small engineering notes illustrate how fast it is to ingest and query short documents.

## Highlights
- Minimal ingestion time (two files totalling a few kilobytes).
- Great for sanity-checking your environment or demoing the CLI.
- Demonstrates contrast between RAG-augmented answers and baseline LLM output.

## Contents
- `files/` – Markdown + text notes about neural scaling laws and engineering best practices.
- `llamafarm-example-quick.yaml` – lightweight config with a simple processor.
- `update_config.sh` – copies the config into place.
- `run_example.sh` – interactive walkthrough using the latest CLI commands.

## Quickstart
```bash
lf init my-quick-demo
/path/to/llamafarm/examples/quick-rag/update_config.sh /path/to/your/project
/path/to/llamafarm/examples/quick-rag/run_example.sh /path/to/your/project
```
Set `NO_PAUSE=1` to skip the “press Enter” prompts.

## Manual Workflow
```bash
lf datasets create -s quick_note_processor -b quick_rag_db quick_dataset
lf datasets upload quick_dataset examples/quick-rag/files/*
lf datasets process quick_dataset

# Retrieval & chat
lf rag query --database quick_rag_db --top-k 3 --include-metadata --include-score \
  "Reference material that discusses neural scaling laws."
lf chat --database quick_rag_db "Summarize neural scaling laws in two sentences with citations."
lf chat --no-rag "Summarize neural scaling laws in two sentences."
```

## Cleanup
```bash
lf datasets delete quick_dataset
rm -rf data/quick_rag_db
```
