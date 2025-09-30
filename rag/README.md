# RAG Toolkit

Python helpers used by the server and CLI for ingestion, parsing, and retrieval. Day-to-day usage is through the `lf` CLI—this package exists so you can run workers/tests locally and extend the pipeline.

## Run the Celery Worker (development)
```bash
uv sync
uv run python cli.py worker
```

> You normally don’t need to start this manually. `lf start` and `lf datasets process` will launch a worker via Docker automatically.

### Starting Services with Nx (from repo root)
```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# All-in-one
nx dev

# Separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

## CLI Utilities (Advanced)
There is still a thin Python CLI in `cli.py` for low-level debugging, but the recommended interface is the Go-based `lf` command. Use the Go CLI for:
- Creating/uploading/processing datasets (`lf datasets …`)
- Querying documents (`lf rag query …`)
- Managing health/stats (`lf rag health|stats`)

## Configuration
The ingestion pipeline is driven by the `rag` section of `llamafarm.yaml` (see `../config/schema.yaml` and `../docs/website/docs/rag/index.md`). Key concepts:
- **Databases** (`ChromaStore`, `QdrantStore`, …) with embedding/retrieval strategy definitions.
- **Data processing strategies** specifying parsers, extractors, metadata processors.
- **Datasets** referencing a processing strategy + database pair.

Update `rag/schema.yaml` when adding new parsers, extractors, or stores, then regenerate types via `config/generate-types.sh`.

## Tests
```bash
uv run python cli.py test   # Smoke tests for strategies
uv run pytest tests/
```

Those ensure strategies defined in the templates and schema remain valid.

## Documentation
User-facing instructions for ingestion, queries, and extending RAG live in `docs/website/docs/rag/index.md`.
