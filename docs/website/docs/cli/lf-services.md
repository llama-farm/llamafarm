---
title: lf services
sidebar_position: 9
---

# lf services

Manage LlamaFarm backend services (server, RAG worker, universal runtime).

## Synopsis

```bash
lf services <subcommand> [service-name] [flags]
```

## Subcommands

| Command | Description |
|---------|-------------|
| `status` | Check status of all services |
| `start` | Start services |
| `stop` | Stop services |

## Available Services

| Service | Description | Default Port |
|---------|-------------|--------------|
| `server` | Main FastAPI server | 8000 |
| `rag` | RAG/Celery worker | N/A |
| `universal-runtime` | Universal Runtime server for HuggingFace models | 11540 |

## Orchestration Modes

LlamaFarm supports multiple orchestration modes, controlled by the `LF_ORCHESTRATION_MODE` environment variable:

| Mode | Description |
|------|-------------|
| `native` (default) | Run services as native processes |
| `docker` | Run services as Docker containers |
| `auto` | Auto-detect best mode (prefers native) |

## lf services status

Check the current status of all LlamaFarm services without starting them.

```bash
lf services status [flags]
```

### Flags

| Flag | Description |
|------|-------------|
| `--json` | Output status in JSON format |

### Output Information

The status command shows:
- Process/container running state
- PID (native) or container ID (Docker)
- Port mappings (Docker)
- Health status (if service is running)
- Log file location (native) or image information (Docker)
- Uptime

### Examples

```bash
# Check status of all services
lf services status

# Get machine-readable JSON output
lf services status --json
```

### Sample Output

```
LlamaFarm Services Status
═════════════════════════

  Service             State      PID      Health    Uptime
  ──────────────────────────────────────────────────────────
  server              running    12345    healthy   2h 15m
  rag                 running    12346    healthy   2h 15m
  universal-runtime   stopped    -        -         -
```

## lf services start

Start LlamaFarm services.

```bash
lf services start [service-name] [flags]
```

### Arguments

- `service-name` (optional): Specific service to start. If omitted, starts all services.

### Examples

```bash
# Start all services
lf services start

# Start only the main server
lf services start server

# Start only the RAG worker
lf services start rag

# Start only the Universal Runtime
lf services start universal-runtime

# Start all services using Docker
LF_ORCHESTRATION_MODE=docker lf services start
```

## lf services stop

Stop LlamaFarm services.

```bash
lf services stop [service-name] [flags]
```

### Arguments

- `service-name` (optional): Specific service to stop. If omitted, stops all services.

### Examples

```bash
# Stop all services
lf services stop

# Stop only the main server
lf services stop server

# Stop only the RAG worker
lf services stop rag

# Stop all Docker containers
LF_ORCHESTRATION_MODE=docker lf services stop
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LF_ORCHESTRATION_MODE` | Orchestration mode (`native`, `docker`, `auto`) | `native` |

## Use Cases

### Development Workflow

```bash
# Start all services for development
lf services start

# Check everything is running
lf services status

# When done, stop everything
lf services stop
```

### Debugging Service Issues

```bash
# Check which services are running
lf services status --json | jq '.services[] | select(.state == "running")'

# Restart a specific service
lf services stop server
lf services start server
```

### Docker Deployment

```bash
# Start services in Docker mode
LF_ORCHESTRATION_MODE=docker lf services start

# Check container status
lf services status

# Stop Docker containers
LF_ORCHESTRATION_MODE=docker lf services stop
```

## Troubleshooting

### Service Won't Start

1. Check if the port is already in use:
   ```bash
   lsof -i :8000  # For server
   lsof -i :11540 # For universal-runtime
   ```

2. Kill conflicting processes:
   ```bash
   kill <PID>
   ```

3. Try starting the service again:
   ```bash
   lf services start server
   ```

### Service Shows "unhealthy"

1. Check the service logs (native mode):
   ```bash
   # Log files are shown in status output
   tail -f ~/.llamafarm/logs/server.log
   ```

2. Verify dependencies are running:
   - Server needs Redis for Celery
   - RAG worker needs ChromaDB
   - Universal Runtime needs sufficient memory for models

### Docker Mode Issues

1. Ensure Docker is running:
   ```bash
   docker ps
   ```

2. Check Docker container logs:
   ```bash
   docker logs llamafarm-server
   ```

## See Also

- [lf start](./lf-start.md) - Start server and open interactive chat
- [lf init](./lf-init.md) - Initialize a new project
- [Configuration Guide](../configuration/index.md) - Configure services
