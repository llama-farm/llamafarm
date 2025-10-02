---
title: lf start
sidebar_position: 2
---

# `lf start`

Launch the local development stack (API server + RAG worker) and open an interactive chat UI with live config watching.

## Synopsis

```
lf start [flags]
```

## Flags

| Flag | Description |
| ---- | ----------- |
| `--ollama-host` | Override the Ollama endpoint (default `http://localhost:11434`). |
| Global flags | `--server-url`, `--debug`, `--cwd`, etc. |

Environment variables:
- `OLLAMA_HOST` – fallback if the flag is not provided.

## Behaviour

- Ensures the server and RAG worker containers are running (via the container orchestrator/Nx).
- Starts a config watcher so edits to `llamafarm.yaml` refresh automatically.
- Launches a text UI for chatting with the current project; use `Ctrl+C` to exit.
- Shows health diagnostics for server, storage, Ollama/vLLM, Celery, rag-service, and project state.

## Example

```bash
lf start --ollama-host http://localhost:11434
```

**Typical health banner**

```
⚠️  Server is degraded
Summary: server=healthy, storage=healthy, ollama=healthy, celery=degraded, rag-service=healthy, project=healthy
  ⚠️ celery  degraded   No workers replied to ping (latency: 533ms)
```

Investigate degraded components before shipping to production—Celery workers might be offline or overloaded.

## See Also

- [`lf chat`](./lf-chat.md)
- [Troubleshooting](../troubleshooting/index.md)
