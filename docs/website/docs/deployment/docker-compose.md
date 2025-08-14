---
title: Docker Compose
sidebar_label: Docker Compose
slug: /deployment/docker-compose
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Run LlamaFarm quickly with Docker or Compose.

## Prerequisites

- Docker Engine
- Docker Compose plugin

## Build and run

```bash
# Build image
llamafarm build --docker

# Run container
docker run -p 8080:8080 llamafarm:latest

# With GPU support
docker run --gpus all -p 8080:8080 llamafarm:latest
```

## Compose file (layout)

Use the provided `docker-compose.yml` in the repo for a multi-service setup (API, workers, vector DB). Adjust ports, volumes, and GPU settings as needed.

## Start/stop and logs

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```
