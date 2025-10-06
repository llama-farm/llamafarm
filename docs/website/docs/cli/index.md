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

Environment helpers:
- `LLAMAFARM_SESSION_ID` – reuse a session for `lf chat`.
- `OLLAMA_HOST` – point `lf start` to a different Ollama endpoint.

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
| [`lf version`](./lf-version.md) | Print CLI version/build info. |

## Troubleshooting CLI Output

- **“Server is degraded”** – At least one dependency (Celery, RAG worker, Ollama) is slow or offline. Commands may still succeed; check logs if they hang.
- **“No response received”** – The runtime streamed nothing; run with `--no-rag` or change models if the provider struggles with tools output.
- **Dataset processing timeouts** – The CLI times out after waiting for Celery. Re-run once ingestion finishes or increase worker availability.
- **Authorization redaction** – `lf chat --curl` hides API keys automatically; replace `<redacted>` before running.

Looking to add a new command? See [Extending LlamaFarm](../extending/index.md#extend-the-cli) for a Cobra walkthrough.
