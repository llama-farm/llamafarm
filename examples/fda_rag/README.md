# FDA Correspondence RAG Example

This example ingests a small set of publicly available FDA correspondence letters and demonstrates how to search and chat over them with LlamaFarm.

## Prerequisites

- LlamaFarm CLI built (`go build -o lf ./cli`)
- Local server + Ollama stack running (e.g. `./start-local.sh` or individual services)

## Quickstart

```bash
# 1. Optionally initialise a fresh project directory
mkdir -p ~/projects/fda-demo && cd ~/projects/fda-demo
lf init  # or ./lf init if running from repo root

# 2. Apply the example configuration (backs up the current config)
/path/to/llamafarm/examples/fda_rag/update_config.sh /path/to/your/project

# 3. Run the full ingestion + search + chat workflow
/path/to/llamafarm/examples/fda_rag/run_example.sh /path/to/your/project
```

> Both scripts default to the repository root if no path is supplied. Pass an explicit project path if you keep your config elsewhere.

## Sample FDA letters included

The three PDFs used in this demo ship with the example in `examples/fda_rag/files/`:

- `761225_2024_Orig1s000OtherActionLtrs.pdf`
- `761240_2023_Orig1s000OtherActionLtrs.pdf`
- `761248_2024_Orig1s000OtherActionLtrs.pdf`

## What the scripts do

- **`update_config.sh`** – backs up the existing `llamafarm.yaml` (if present) and copies `llamafarm-example-fda.yaml` into place.
- **`run_example.sh`** –
  1. Ensures the dataset exists (recreates it if necessary)
  2. Ingests the three FDA PDF letters into the configured database
  3. Displays dataset metadata
  4. Processes the dataset into the vector store
  5. Performs an `lf rag query` search to preview retrieved passages
  6. Runs `lf run` twice with RAG enabled (and once with `--no-rag`) to show contrast between augmented vs LLM-only answers (the script warns if a command fails so you can retry manually)

Both scripts emit coloured log lines so you can follow progress, and any failures are echoed so you can rerun the exact command.

## Manual workflow (alternative)

If you prefer to run the commands yourself:

```bash
./lf datasets add fda_letters -s fda_pdf_processor -b fda_letters_db
./lf datasets ingest fda_letters examples/fda_rag/files/*.pdf
./lf datasets process fda_letters
./lf datasets list

# Preview retrieved passages
./lf rag query --database fda_letters_db --top-k 3 --include-metadata --include-score \
  "Which FDA letters mention additional clinical trial data requirements?"

# Ask questions with RAG
./lf run --database fda_letters_db \
  "Summarize the key deficiencies highlighted across the 2024 FDA correspondence letters."
./lf run --database fda_letters_db \
  "According to FDA correspondence 761240, what follow-up actions were requested from the sponsor?"
./lf run --database fda_letters_db \
  "What timeline-related requests are mentioned for sponsors in the 2024 FDA letters?"

# Baseline without RAG
./lf run --no-rag \
  "According to FDA correspondence 761240, what follow-up actions were requested from the sponsor?"

# Additional prompts once processing finishes
./lf rag query --database fda_letters_db --top-k 5 \
  "Which letters include requests for additional safety analyses?"
./lf rag query --database fda_letters_db --top-k 5 \
  "Find correspondence that mentions manufacturing deficiencies."
./lf run --database fda_letters_db \
  "List outstanding deliverables cited by the FDA across the 2024 letters, including citations."
```

## Cleanup

```bash
./lf datasets remove fda_letters
rm -rf data/fda_letters_db
```

Restore your original configuration from the backup created by `update_config.sh` if needed.
