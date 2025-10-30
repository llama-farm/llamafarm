---
title: Docker Compose
slug: /deployment/docker-compose
---

# Docker Compose

Run the server and RAG worker with Docker Compose for reproducible environments.

## Prerequisites

- Docker Engine + Compose plugin
- Built artifacts or Dockerfiles in `server/` and `rag/`
- A project config mounted into the containers (e.g., `llamafarm.yaml`)

## Example Compose File

```yaml
version: "3.8"
services:
  server:
    build: ./server
    environment:
      - LLAMAFARM_CONFIG=/app/llamafarm.yaml
      - RUNTIME_API_KEY=${RUNTIME_API_KEY}
    volumes:
      - ../llamafarm.yaml:/app/llamafarm.yaml:ro
    ports:
      - "8000:8000"

  rag-worker:
    build: ./rag
    environment:
      - LLAMAFARM_CONFIG=/app/llamafarm.yaml
    volumes:
      - ../llamafarm.yaml:/app/llamafarm.yaml:ro
    depends_on:
      - server

  designer:
    image: ghcr.io/llama-farm/llamafarm/designer:latest
    ports:
      - "3123:80"
    environment:
      - VITE_APP_API_URL=http://server:8000
    depends_on:
      - server
```

Adjust paths according to your directory layout. Mount datasets or persistent volumes if you need durable storage for vector databases.

The Designer web UI will be available at `http://localhost:3123` and provides a visual interface for managing projects, datasets, and configurations. See the [Designer documentation](../designer/index.md) for details.

## Usage

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Tail logs
docker compose logs -f server

# Stop services
docker compose down
```

If you rely on Ollama or another runtime, run it on the host or add a service to the Compose file and set `runtime.base_url` accordingly.

## Tips

- Inject secrets via `.env` files or Docker secrets; never bake API keys into images.
- Use healthchecks in Compose to ensure workers wait for the server to become ready.
- Scale workers with `docker compose up --scale rag-worker=3` when ingestion throughput matters.

For Kubernetes and other orchestrators, see [Deployment](./index.md).
