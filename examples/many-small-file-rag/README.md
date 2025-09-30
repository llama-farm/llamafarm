# Many Small Files RAG Example (FDA Letters)

Demonstrates ingesting several short FDA correspondence PDFs to show a quick multi-document workflow.

## Focus Areas
- Uploading multiple 10–15 page letters rather than a single large tome.
- Quick processing, making it ideal for iterative testing.
- Citation-rich prompts referencing letter identifiers.

## Contents
- `files/` – FDA correspondence letters (PDFs).
- `llamafarm.yaml` – configuration tuned for FDA processing, picked up automatically via `lf --cwd`.
- `run_example.sh` – interactive workflow using `lf datasets`, `lf rag`, and `lf chat`.

## Quickstart
```bash
# From the repo root (expects ./lf or export LF_BIN=/full/path/to/lf)
./examples/many-small-file-rag/run_example.sh

# Optional: run against a different checkout/build directory that contains the lf binary
./examples/many-small-file-rag/run_example.sh /path/to/your/project
```
Use `NO_PAUSE=1` to skip interactive prompts.

The script duplicates the FDA database definition with a unique name, creates a fresh dataset, uploads the letters, processes them, and issues retrieval/chat commands that cite individual correspondence IDs.

## Manual Workflow
```bash
lf --cwd examples/many-small-file-rag datasets create -s fda_pdf_processor -b fda_letters_db fda_letters
lf --cwd examples/many-small-file-rag datasets upload fda_letters examples/many-small-file-rag/files/*.pdf
lf --cwd examples/many-small-file-rag datasets process fda_letters

lf --cwd examples/many-small-file-rag rag query --database fda_letters_db --top-k 3 --include-metadata --include-score \
  "Which FDA letters mention additional clinical trial data requirements?"

lf --cwd examples/many-small-file-rag chat --database fda_letters_db "Summarize key deficiencies highlighted across the 2024 FDA letters with citations."
lf --cwd examples/many-small-file-rag chat --database fda_letters_db "According to correspondence 761240, what follow-up actions were requested?"
lf --cwd examples/many-small-file-rag chat --no-rag "According to correspondence 761240, what follow-up actions were requested?"
```

## Cleanup
```bash
lf --cwd examples/many-small-file-rag datasets delete fda_letters
rm -rf examples/many-small-file-rag/data/fda_letters_db
```
