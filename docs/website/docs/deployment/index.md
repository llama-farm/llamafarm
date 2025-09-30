---
title: Deployment
sidebar_position: 9
---

# Deployment

LlamaFarm shards into two primary services: the FastAPI server and the RAG worker (Celery). You can run them locally for development or deploy to your infrastructure using the same configuration.

## Local Development

### Option A: `lf start`

The easiest path is the integrated dev stack:

```bash
lf start
```

This command:
- Starts the server and RAG worker (using Docker/Nx under the hood).
- Watches `llamafarm.yaml` for changes.
- Opens an interactive chat UI.

### Option B: Manual control

Open two terminals:

```bash
# Terminal 1 – API
cd server
uv sync
uv run uvicorn server.main:app --reload

# Terminal 2 – RAG worker
cd rag
uv sync
uv run python cli.py worker
```

Then run CLI commands against the default server at `http://localhost:8000`.

## Production Checklist

- **Environment variables**: store API keys (OpenAI, Together, etc.) in `.env` files or secret managers. Update `runtime.api_key` to reference them.
- **Process management**: use process supervisors (systemd, PM2, Docker Compose) to keep server and Celery workers running.
- **Workers**: size Celery workers based on ingestion load; large PDFs can take minutes.
- **Observability**: enable logging and monitoring around ingestion jobs (out-of-scope for this quick guide, but recommended).

## Docker Compose (Example)

```yaml
docker-compose.yml
version: "3.8"
services:
  server:
    build: ./server
    environment:
      - LLAMAFARM_CONFIG=/app/llamafarm.yaml
      - RUNTIME_API_KEY=${RUNTIME_API_KEY}
    volumes:
      - ./llamafarm.yaml:/app/llamafarm.yaml
    ports:
      - "8000:8000"
  rag-worker:
    build: ./rag
    environment:
      - LLAMAFARM_CONFIG=/app/llamafarm.yaml
    volumes:
      - ./llamafarm.yaml:/app/llamafarm.yaml
```

Adjust to mount datasets, persistence layers, or external vector stores as needed.

## Scaling Beyond Local

- **vLLM / Hosted runtimes** – run models on separate infrastructure; update `runtime.base_url` and `api_key`.
- **Managed vector stores** – swap `ChromaStore` for Qdrant cloud or another backend you register.
- **Multiple workers** – run additional Celery workers to parallelize ingestion.
- **Container orchestration** – convert the Docker Compose example to Kubernetes or ECS, ensuring environment variables and secrets propagate.

## Tests Before Release

- `uv run --group test python -m pytest` (server)
- `go test ./...` (CLI)
- `nx build docs` (documentation)
- Smoke tests on ingestion/queries using `lf datasets` and `lf rag query`

## Resources

- [Quickstart](../quickstart/index.md) – local installation steps.
- [Configuration Guide](../configuration/index.md) – runtime/provider settings.
- [Extending LlamaFarm](../extending/index.md) – adapt to your infrastructure.
