# LlamaFarm Server

FastAPI application that powers project chat, dataset APIs, and health checks.

## Running Locally
The CLI (`lf start`) will launch the server and RAG worker for you, but you can run it manually while developing inside `server/`.

```bash
uv sync
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

To execute Celery ingestion jobs alongside it, start the worker from `rag/` (see that README) or run `lf datasets process …` which will auto-start the worker via Docker.

Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Running via Nx from the Repository Root
If you prefer to use the same orchestration as the CLI without Docker auto-start:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Option A: single command
nx dev

# Option B: separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

## Tests
```bash
uv run --group test python -m pytest
```

## Configuration
The server reads `llamafarm.yaml` via the config package. Ensure your project config includes:
- `runtime` with provider/model/base_url/api_key as required.
- `rag` strategies/databases that match datasets you plan to ingest.

For a complete schema reference and instructions on extending endpoints, see the main documentation at `docs/website/docs`.
