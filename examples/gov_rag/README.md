# Raleigh Unified Development Ordinance (UDO) RAG Example

This example ingests the Raleigh UDO Supplement 31 PDF and demonstrates how to query zoning regulations and obtain cited responses.

## Prerequisites
- LlamaFarm CLI built (`go build -o lf ./cli`)
- Local server, Celery/RAG worker, and Ollama running (`./start-local.sh` or run `nx start server`, `nx start rag`, and `ollama serve`)

## Quickstart
```bash
# (Optional) create a fresh project directory
mkdir -p ~/projects/raleigh-udo && cd ~/projects/raleigh-udo
lf init

# Apply the example configuration (backs up your current config)
/path/to/llamafarm/examples/gov_rag/update_config.sh /path/to/your/project

# Run ingestion + RAG walkthrough (script exits early if the RAG worker is offline)
/path/to/llamafarm/examples/gov_rag/run_example.sh /path/to/your/project
```
> Both scripts default to the repo root if no path argument is supplied.

## Included Document
- `examples/gov_rag/files/UDOSupplement31.pdf`

## What the scripts do
- **`update_config.sh`** – backs up `llamafarm.yaml` and installs `llamafarm-example-gov.yaml` with:
  - Custom prompt requiring section citations
  - PDF parsing tuned for ordinance documents (semantic chunking + heading extraction)
- **`run_example.sh`** –
  1. Recreates the dataset `raleigh_udo_dataset`
  2. Uploads the UDO PDF
  3. Processes it into the database `raleigh_udo_db` (processing may take several minutes; check rag-worker logs if it appears to stall)
  4. Runs a semantic `lf rag query`
  5. Executes multiple RAG-enabled chats plus an LLM-only baseline for comparison (the script warns if a command fails so you can retry manually)

## Useful commands
```bash
# Manually ingest and process
./lf datasets add raleigh_udo_dataset -s udo_pdf_processor -b raleigh_udo_db
./lf datasets ingest raleigh_udo_dataset examples/gov_rag/files/UDOSupplement31.pdf
./lf datasets process raleigh_udo_dataset

# Targeted queries
./lf rag query --database raleigh_udo_db --top-k 3 \
  "Where does the UDO discuss neighborhood transition requirements?"
./lf run --database raleigh_udo_db \
  "Summarize Village Mixed Use height limits and cite the section."
./lf run --database raleigh_udo_db \
  "Detail buffering requirements when commercial uses adjoin residential lots."

# Baseline (no RAG)
./lf run --no-rag "How tall can buildings be in Village Mixed Use zoning?"

# Additional prompts once processing finishes (replace with your own projects/questions)
./lf rag query --database raleigh_udo_db --top-k 5 \
  "Provide sections covering neighborhood conservation overlay districts and their intent."
./lf rag query --database raleigh_udo_db --top-k 5 \
  "Where does the ordinance require streetscape planting or street trees for mixed use projects?"
./lf run --database raleigh_udo_db \
  "Detail the parking reductions permitted in Transit Overlay Districts and cite the relevant subsections."
./lf run --database raleigh_udo_db \
  "What special use permit findings must City Council make for master plan approvals? Include citations."
```

## Cleanup
```bash
./lf datasets remove raleigh_udo_dataset
rm -rf data/raleigh_udo_db
```
Restore your previous `llamafarm.yaml` from the backup created by `update_config.sh` if needed.
