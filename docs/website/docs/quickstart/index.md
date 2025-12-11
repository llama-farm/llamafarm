---
title: Quickstart
sidebar_position: 1
---

# Quickstart

Get the CLI installed, ingest a dataset, and run your first RAG-powered chat in minutes.

## 1. Prerequisites

- [Docker](https://www.docker.com/get-started/)
- [Ollama](https://ollama.com/download) _(local runtime today; additional providers coming soon)_

> Docker is used to run the API and RAG worker automatically when you invoke `lf start`.

## 2. Install the CLI

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

**Windows/Linux Desktop App:**

| Platform | Download |
|----------|----------|
| **Windows** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/download/v0.0.20/LlamaFarm.Setup.0.0.20.exe) |
| **Linux** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/download/v0.0.20/LlamaFarm-0.0.20.AppImage) |
| **Mac (M1+)** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/download/v0.0.20/LlamaFarm-0.0.20-arm64-mac.zip) |

Or download the CLI only: grab `lf.exe` (Windows) or `lf` binary from the [releases page](https://github.com/llama-farm/llamafarm/releases/latest) and add it to your PATH.

Confirm everything is wired up:

```bash
lf --help
```

## 3. Tune Your Runtime (Ollama)

For best RAG results with longer documents, increase the Ollama context window to match production expectations (e.g., 100K tokens):

1. Open the Ollama app.
2. Navigate to **Settings → Advanced**.
3. Adjust the context window to your desired size.

## 4. Create a Project

```bash
lf init my-project
```

This reaches the server (auto-started if needed) and writes `llamafarm.yaml` with default runtime, prompts, and RAG configuration.

## 5. Start the Local Stack

```bash
lf start
```

- Spins up the FastAPI server and RAG worker via Docker.
- Starts a config watcher and opens the interactive dev chat TUI.
- Shows health diagnostics for Ollama, Celery, and the rag-service.
- Launches the Designer web UI at `http://localhost:8000` for visual project management.

Hit `Ctrl+C` to exit the chat UI when you're done.

:::tip Use the Designer Web UI
Prefer a visual interface? Open `http://localhost:8000` in your browser to access the Designer, where you can manage projects, upload datasets, configure models, and test prompts—all without touching the command line.

See the [Designer documentation](../designer/index.md) for details.
:::

### Running Services Manually (no Docker auto-start)

If you want to control each service yourself (useful when hacking on code), launch them with Nx from the repository root:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Option A: start both services together
nx dev

# Option B: start in separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

Open another terminal to run `lf` commands against the locally running stack.

## 6. Chat with Your Project

```bash
# Interactive chat (opens TUI using project from llamafarm.yaml)
lf chat

# One-off message
lf chat "What can you do?"
```

Options you'll likely use:

- `--no-rag` – bypass retrieval and hit the runtime directly.
- `--database`, `--retrieval-strategy` – override RAG behaviour.
- `--curl` – print the sanitized `curl` command instead of executing.

## 7. Create and Populate a Dataset

```bash
# Create dataset with configured strategy/database
lf datasets create -s pdf_ingest -b main_db research-notes

# Upload documents (supports globs/directories)
lf datasets upload research-notes ./examples/fda_rag/files/*.pdf
```

The CLI validates strategy and database names against your `rag` configuration and reports upload successes/failures.

## 8. Process Documents

```bash
lf datasets process research-notes
```

- Sends an ingestion job to Celery.
- Shows heartbeat dots (TTY only) so long-running jobs feel alive.
- For large PDFs, the worker may need extra time—rerun the command if you see a timeout message.

## 9. Query with RAG

```bash
lf rag query --database main_db "Which FDA letters mention clinical trial data?"
```

Useful flags: `--top-k 10`, `--filter "file_type:pdf"`, `--include-metadata`, `--include-score`.

## 10. Reset Sessions (Optional)

For stateless testing, clear dev history by removing `.llamafarm/projects/<namespace>/<project>/dev/context`, or start a new namespace/project.

## 11. Next Steps

- [Designer Web UI](../designer/index.md) — visual interface for managing projects.
- [Configuration Guide](../configuration/index.md) — deep dive into `llamafarm.yaml`.
- [RAG Guide](../rag/index.md) — strategies, parsers, and retrieval.
- [Extending LlamaFarm](../extending/index.md) — add new providers, stores, or parsers.
- [Examples](../examples/index.md) — run the FDA and Raleigh demos end-to-end.

Need help? Chat with us on [Discord](https://discord.gg/RrAUXTCVNF) or open a [discussion](https://github.com/llama-farm/llamafarm/discussions).
