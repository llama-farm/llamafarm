---
title: Troubleshooting & FAQ
sidebar_position: 11
---

# Troubleshooting & FAQ

Common symptoms, their causes, and how to resolve them when working with LlamaFarm.

## CLI & Chat

| Issue | Cause | Fix |
| ----- | ----- | --- |
| `Server is degraded` banner | A dependency (Celery, rag-service, Ollama) is slow or offline. | Ensure services are running, restart `lf start`, inspect logs. |
| Service unhealthy (check with `lf services status`) | One or more services failed to start or crashed. | Run `lf services stop`, then `lf services start` to restart. Check logs for errors. |
| `No response received` | Runtime returned an empty stream (model/tool mismatch). | Use `--no-rag`, switch to an agent handler the model supports, or choose a different model. |
| `InstructorRetryException: ... does not support tools` | You selected structured output but the model lacks tool support. | Set `instructor_mode: null` and use simple chat, or choose a capable model. |
| `context deadline exceeded` during queries | Long-running HTTP request or server busy. | Increase timeout, retry after the worker finishes, or scale workers. |

## Dataset Processing

| Issue | Cause | Fix |
| ----- | ----- | --- |
| `Task timed out or failed: PENDING` | Celery worker still ingesting large files. | Wait, then rerun `lf datasets process`; monitor worker logs. |
| Duplicates skipped | File already processed (hash matches). | Remove the file or ingest new content; duplicates are safe to ignore. |
| Metadata mismatch warnings | Strategy name or database not defined in config. | Update `llamafarm.yaml` to include the strategy/database. |

## Configuration Errors

| Error | Fix |
| ----- | --- |
| `runtime.provider is not one of ...` | Update provider enum or choose a supported value. |
| `Missing required property runtime.base_url` | Provide `base_url` when using non-default provider endpoints (vLLM, Together). |
| `llamafarm.yaml not found` | Run `lf init` or set `--cwd` to a directory containing the config. |

## Installation Issues

| Issue | Cause | Fix |
| ----- | ----- | --- |
| `lf` command not found after installation | Binary not in system PATH | Ensure `/usr/local/bin` (or custom install directory) is in your PATH. Run `echo $PATH` to verify. |
| Installation script fails | Permissions issue or unsupported platform | Try with `sudo` or check platform compatibility (macOS/Linux supported). Windows users should download `lf.exe` manually. |

## Sessions & State

- Delete `.llamafarm/projects/<namespace>/<project>/dev/context` to reset dev chat history.
- Use a new namespace (`lf init --namespace new-team`) for isolated experiments.
- Pass explicit `session_id` to API calls or set `LLAMAFARM_SESSION_ID` when testing stateless flows.

## Build & Development Issues

| Issue | Cause | Fix |
| ----- | ----- | --- |
| `nx build docs` sqlite I/O error | Nx cache database lock or permission issue | Remove `.nx/` cache directory or run with `NX_SKIP_NX_CACHE=1`. |
| Docker containers fail to start | Port conflicts or missing dependencies | Check if ports 8000/6379/5432 are available. Run `lf services stop` first, then `lf services start`. |
| Python module import errors | Virtual environment not activated or dependencies not installed | Run `cd server && uv sync` or `cd rag && uv sync` to install dependencies. |

## Extensibility Pitfalls

- Forgot to regenerate types after editing `config/schema.yaml` or `rag/schema.yaml` → run `config/generate_types.py`.
- Added a provider/store without updating docs → document how to configure it so others know it exists.
- CLI command not appearing → ensure you added it via `rootCmd.AddCommand()` and compiled with `go build`.

Still stuck? Ask in [Discord](https://discord.gg/RrAUXTCVNF) or create a [discussion](https://github.com/llama-farm/llamafarm/discussions).
