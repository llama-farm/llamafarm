---
title: lf debug logs
sidebar_position: 6
---

# `lf debug logs`

Stream or snapshot logs from your **local LlamaFarm development stack** — the containers started by `lf start`.

Use this command to inspect logs for services like the **server**, **RAG worker**, or **frontend**, either as a static snapshot or a live tail.

---

## Usage

```bash
lf debug logs [flags]

# Show last 200 lines from all services
lf debug logs

# Live tail all logs
lf debug logs -f

# Only show server and RAG services, tail 1000 lines
lf debug logs -s server -s rag -n 1000

# Show logs since 15 minutes ago
lf debug logs --since 15m

# Stream logs and save output to a file
lf debug logs -f -o .llamafarm/logs/dev-$(date +%Y%m%d-%H%M%S).log

# Use a custom compose file
lf debug logs --compose-file ./deployment/docker_compose/docker-compose.yml

```bash