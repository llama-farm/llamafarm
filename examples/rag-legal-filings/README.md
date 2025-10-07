# RAG example for James Bond legal filings

Process real legal filings from James Bond franchise litigation — including MGM v. American Honda Motor, Danjaq LLC v. McClory, and other 007-related court cases — and explore how RAG handles complex legal documents with citations and cross-references.

## What This Demo Highlights
- Parsing and chunking legal PDF filings with multiple extractors optimized for legal content.
- Creating isolated databases/datasets per run to avoid stale state.
- Example questions that require legal citations and cross-case comparisons.
- Contrast between RAG-augmented responses and baseline LLM output for legal queries.
- Entity extraction for organizations, dates, locations, and legal facilities.

## Directory Contents
- `llamafarm.yaml` – configuration tuned for legal PDF processing with specialized system prompt for legal assistant role.
- `run_example.sh` – interactive workflow using the latest CLI commands (`lf chat`, `lf datasets`, etc.).

## Quickstart
```bash
# From the repo root (expects ./lf or export LF_BIN=/full/path/to/lf)
./examples/007/run_example.sh

# Optional: run against a different checkout/build directory that contains the lf binary
./examples/007/run_example.sh /path/to/your/project
```
Set `NO_PAUSE=1` to skip prompts during automated runs.

The script clones a base database strategy, creates unique database/dataset names, uploads legal PDFs, processes them with legal-specific extractors, runs retrieval, and issues several `lf chat` prompts with and without RAG. Each command's output is printed so you can inspect chunk counts, legal metadata, and responses with proper citations.

## Manual Workflow
```bash
# Create a dataset and point to the 007 legal filings PDF processor
lf --cwd examples/007 datasets create -s 007_legal_filings_pdf_processor -b 007_legal_filings_db 007_legal_filings_dataset

# Upload the legal filing PDFs
lf --cwd examples/007 datasets upload 007_legal_filings_dataset *.pdf

# Process and inspect
lf --cwd examples/007 datasets process 007_legal_filings_dataset
lf --cwd examples/007 datasets list
lf --cwd examples/007 rag query --top-k 3 --include-metadata --include-score \
  "What was the main dispute in MGM v. American Honda Motor?"

# Ask questions with RAG
lf --cwd examples/007 chat "Summarize the key legal issues in the Danjaq LLC v. McClory case with citations."
lf --cwd examples/007 chat "List the parties involved in the Universal City Studios litigation."

# Baseline comparison
lf --cwd examples/007 chat --no-rag "What legal issues has the James Bond franchise faced?"
```

## Sample Questions to Try
- "What was the outcome of MGM v. American Honda Motor and what precedent did it set?"
- "How did the courts rule on the use of James Bond characters in advertising?"
- "What are the key differences between the Danjaq and Universal City Studios cases?"
- "Which legal filings mention specific James Bond films or characters?"
- "What damages were sought in these trademark disputes?"

## Cleanup
```bash
lf --cwd examples/007 datasets delete 007_legal_filings_dataset
rm -rf examples/007/data/007_legal_filings_db
```
