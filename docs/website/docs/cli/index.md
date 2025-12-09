---
title: CLI Reference
sidebar_position: 3
---

# CLI Reference

The `lf` CLI is your control center for LlamaFarm projects. This reference captures global flags, command behaviours, and examples you can copy into your shell. Each subcommand shares the same auto-start logic: if the server or RAG worker is not running locally, the CLI will launch them (unless you override `--server-url`).

## Global Flags

```
lf [command] [flags]
```

| Flag | Description |
| ---- | ----------- |
| `--debug`, `-d` | Enable verbose logging. |
| `--server-url` | Override the server endpoint (default `http://localhost:8000`). |
| `--server-start-timeout` | How long to wait for local server startup (default 45s). |
| `--cwd` | Treat another directory as the working project root. |
| `--auto-start` | Automatically start services when needed (default: true). Use `--auto-start=false` to disable. |

Environment helpers:
- `LLAMAFARM_SESSION_ID` – reuse a session for `lf chat`.
- `OLLAMA_HOST` – point `lf start` to a different Ollama endpoint.
- `LF_VERSION_REF` – override the source code version/ref downloaded by the CLI (useful for testing feature branches or specific versions).
- `LF_DATA_DIR` – override the data directory (default: `~/.llamafarm`).

## Command Matrix

| Command | Description |
| ------- | ----------- |
| [`lf init`](./lf-init.md) | Scaffold a project and generate `llamafarm.yaml`. |
| [`lf start`](./lf-start.md) | Launch server + RAG services and open the dev chat UI. |
| [`lf chat`](./lf-chat.md) | Send single prompts, preview REST calls, manage sessions. |
| [`lf models`](./lf-models.md) | List available models and manage multi-model configurations. |
| [`lf datasets`](./lf-datasets.md) | Create, upload, process, and delete datasets. |
| [`lf rag`](./lf-rag.md) | Query documents and access RAG maintenance tools. |
| [`lf projects`](./lf-projects.md) | List projects by namespace. |
| [`lf services`](./lf-services.md) | Manage LlamaFarm services (server, RAG worker, universal runtime). |
| [`lf version`](./lf-version.md) | Print CLI version/build info and check for updates. |

## Service Management with --auto-start

The `--auto-start` flag (default: `true`) gives you control over when services start. By default, the CLI automatically starts the server and RAG worker when needed. Use `--auto-start=false` to:

- **CI/CD pipelines**: Connect to pre-started services without triggering restarts
- **Manual service management**: Keep services running between commands
- **Debugging**: Separate service startup from command execution

### Usage Examples

**Two-terminal development workflow:**
```bash
# Terminal 1: Start and monitor services
lf start

# Terminal 2: Run commands without triggering restarts
lf chat --auto-start=false "What is LlamaFarm?"
lf datasets list --auto-start=false
lf rag stats --auto-start=false
```

**CI/CD integration:**
```yaml
- name: Start services
  run: lf start &
  
- name: Wait for health
  run: timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
  
- name: Run tests
  run: |
    lf datasets create --auto-start=false -s pdf_ingest -b main_db test-data
    lf rag query --auto-start=false "test query"
```

**Error handling:**
If services are not running and you use `--auto-start=false`, you'll see:
```
services not running and auto-start is disabled: server, celery (use --auto-start to enable automatic startup)
```

To start services manually, run:
```bash
lf start
```

Or remove the `--auto-start=false` flag to allow automatic startup.

## Troubleshooting CLI Output

- **"Server is degraded"** – At least one dependency (Celery, RAG worker, Ollama) is slow or offline. Commands may still succeed; check logs if they hang.
- **"No response received"** – The runtime streamed nothing; run with `--no-rag` or change models if the provider struggles with tools output.
- **Dataset processing timeouts** – The CLI times out after waiting for Celery. Re-run once ingestion finishes or increase worker availability.
- **Authorization redaction** – `lf chat --curl` hides API keys automatically; replace `<redacted>` before running.

Looking to add a new command? See [Extending LlamaFarm](../extending/index.md#extend-the-cli) for a Cobra walkthrough.

## Services Management

The `lf services` command provides control over LlamaFarm's backend services.

### Check Service Status

```bash
lf services status
lf services status --json  # Machine-readable output
```

Shows the status of all services (server, RAG worker, universal runtime).

### Start Services

```bash
lf services start              # Start all services
lf services start server       # Start specific service
lf services start rag          # Start RAG worker
lf services start universal-runtime  # Start Universal Runtime
```

**Orchestration Modes:**
- `native` (default): Native processes managed by CLI
- `docker`: Docker containers
- `auto`: Auto-detect best mode

Set via environment variable:
```bash
LF_ORCHESTRATION_MODE=docker lf services start
```

### Stop Services

```bash
lf services stop               # Stop all services
lf services stop server        # Stop specific service
```

### Available Services

| Service | Description | Default Port |
|---------|-------------|--------------|
| `server` | Main FastAPI server | 8000 |
| `rag` | RAG/Celery worker | N/A |
| `universal-runtime` | Universal Runtime server | 11540 |
