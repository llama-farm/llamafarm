---
title: Core Concepts
sidebar_position: 2
---

# Core Concepts

Understand the moving pieces—projects, sessions, runtimes, and the service architecture—before you customize or extend LlamaFarm.

## Architecture Overview

```
┌────────────┐       ┌───────────────┐       ┌──────────────┐
│   lf CLI   │──────▶│  LlamaFarm    │──────▶│ Runtime Host │
│            │ HTTP  │  Server (API) │       │ (Ollama/vLLM │
└─────┬──────┘       │               │       │  OpenAI,...) │
      │              │  ┌─────────┐  │       └──────────────┘
      │ Websocket    │  │ Celery  │◀┐
      │ (streaming)  │  │ Workers │ │ ingest jobs
      │              │  └─────────┘ │
      │              │      ▲       │
      ▼              │      │       │
┌────────────┐       │  ┌─────────┐ │    ┌─────────────┐
│ Config     │◀──────┘  │ RAG     │◀┼────│ Vector Store │
│ Watcher    │ updates   │ Worker  │ │    │ (Chroma,...)│
└────────────┘           └─────────┘ │    └─────────────┘
                                     │
                                     ▼
                               ┌────────────┐
                               │ Dataset    │
                               │ Storage    │
                               └────────────┘
```

- **CLI (`lf`)** orchestrates everything: talking to the API, streaming responses, uploading datasets, and watching config changes.
- **Server** exposes REST endpoints under `/v1/projects/{namespace}/{project}/...` for chat completions, datasets, and RAG queries.
- **Celery workers** handle ingestion tasks asynchronously; the CLI polls and surfaces progress.
- **Runtime hosts** can be local (Ollama) or remote OpenAI-compatible endpoints (vLLM, Together). Configuration controls provider, base URL, API key, and instructor mode.
- **RAG worker** processes documents via configured pipelines and writes to vector databases (default Chroma, configurable).

## Projects & Namespaces

- A **project** is a configuration bundle stored in `llamafarm.yaml` plus server-side metadata.
- Projects live within a **namespace** (defaults to `default`). Namespaces isolate resources, dataset names, and sessions.
- `lf init` creates a project using the server’s template; you can list existing projects with `lf projects list --namespace my-team`.

## Sessions

- `lf chat` creates or resumes a **session** when you pass a `--session-id` or use the environment variable `LLAMAFARM_SESSION_ID`.
- `lf start` opens a stateful dev session whose history persists under `.llamafarm/projects/<namespace>/<project>/dev/context`.
- `lf chat --no-rag` is stateless by default unless you provide a session identifier.
- API consumers pass `session_id` directly to `/chat/completions` to control continuity.

## Configuration-Driven Behaviour

- `llamafarm.yaml` defines runtime, prompts, and RAG strategies (see [Configuration Guide](../configuration/index.md)).
- Changes to the file trigger the config watcher; the CLI reloads live during dev sessions.
- Missing runtime fields (provider/base_url/api_key) are treated as errors; there are no hidden defaults.

## RAG Strategies

- RAG configuration is composed of **databases** and **data processing strategies**.
- Each dataset references a strategy and database; CLI enforces this relationship when creating datasets.
- Strategies describe parsers, extractors, metadata processors, and embedding choices.

## Extensibility Mindset

Everything in LlamaFarm is intended to be swapped or extended:

- Point `runtime.base_url` to a vLLM or custom OpenAI-compatible gateway.
- Register a new vector store backend, update `rag/schema.yaml`, and regenerate types.
- Add parsers/extractors to support new file formats.
- Create new CLI subcommands under `cli/cmd` to automate workflows.

See [Extending LlamaFarm](../extending/index.md) for detailed instructions.

## Component Health

When commands run, you might see a summary like:

```
⚠️ Server is degraded
Summary: server=healthy, storage=healthy, ollama=healthy, celery=degraded, rag-service=healthy, project=healthy
  ⚠️ celery  degraded   No workers replied to ping (latency: 533ms)
```

- **Degraded** does not always mean failure; ingestion may continue in the background.
- `lf rag health` reports live status of embedder, store, and processing pipeline.
- Address warnings before production deployment (ensure Celery workers are running, Ollama/vLLM accessible, etc.).

## Next Steps

- [Quickstart](../quickstart/index.md) – run through the onboarding flow if you haven’t already.
- [CLI Reference](../cli/index.md) – learn each command in detail.
- [RAG Guide](../rag/index.md) – configure databases, strategies, and retrieval.
