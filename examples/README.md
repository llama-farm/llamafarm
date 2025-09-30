# Examples

Sample projects that showcase the `lf` CLI workflows. Full write-ups live in the docs (`docs/website/docs/examples/index.md`).

## Available Folders
- `fda_rag/` – FDA Complete Response letters ingestion + targeted queries.
- `gov_rag/` – Raleigh UDO ingestion (large PDF) with planning queries.

Each folder contains:
- `files/` with source documents.
- `llamafarm-example.yaml` showing a project config tailored to the example.
- `run_all.sh` to run the end-to-end flow (optional helper script).

## Typical Workflow
```bash
# 1. Ensure CLI + prerequisites from the main Quickstart are set
lf init demo

# 2. Overwrite llamafarm.yaml with the example config
cp examples/fda_rag/llamafarm-example.yaml llamafarm.yaml

# 3. Start the services and ingest data
lf start       # or keep an existing stack running in another terminal
lf datasets create -s pdf_ingest -b main_db fda_letters
lf datasets upload fda_letters examples/fda_rag/files/*.pdf
lf datasets process fda_letters

# 4. Query / chat
lf rag query --database main_db "Which letters mention additional clinical trial data?"
lf chat --database main_db "Summarize deficiencies from 2024 letters"
```

Adjust the dataset, strategy, or database names if you customize the config.

## Tips
- Large PDFs may take a few minutes to process; rerun `lf datasets process …` if you see a timeout message—the worker continues ingesting in the background.
- Use `lf rag health` to confirm embedder/store health before querying.
