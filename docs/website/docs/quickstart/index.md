---
title: Quickstart
sidebar_position: 1
---

# Quickstart

Get LlamaFarm installed, ingest a dataset, and run your first RAG-powered chat in minutes.

## 1. Prerequisites

- [Ollama](https://ollama.com/download) — Local model runtime (or any OpenAI-compatible provider)

## 2. Install LlamaFarm

### Option A: Desktop App (Easiest)

Download the all-in-one desktop application:

| Platform | Download |
|----------|----------|
| **Mac (M1+)** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest) |
| **Windows** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest) |
| **Linux** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest) |

The desktop app bundles everything you need—no additional installation required.

### Option B: CLI Installation

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.ps1 | iex
```

**Manual Download:**

Download the `lf` binary directly from the [releases page](https://github.com/llama-farm/llamafarm/releases/latest):

| Platform | Binary |
|----------|--------|
| macOS (Apple Silicon) | `lf-darwin-arm64` |
| macOS (Intel) | `lf-darwin-amd64` |
| Linux (x64) | `lf-linux-amd64` |
| Linux (ARM64) | `lf-linux-arm64` |
| Windows | `lf-windows-amd64.exe` |

After downloading, make it executable and add to your PATH:
```bash
chmod +x lf-darwin-arm64
sudo mv lf-darwin-arm64 /usr/local/bin/lf
```

Verify installation:
```bash
lf --help
```

## 3. Configure Your Runtime (Ollama)

For best RAG results with longer documents, increase the Ollama context window:

1. Open the Ollama app
2. Navigate to **Settings → Advanced**
3. Adjust the context window size (recommended: 32K+ for documents)

Pull a model if you haven't already:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text  # For embeddings
```

## 4. Create a Project

```bash
lf init my-project
cd my-project
```

This creates `llamafarm.yaml` with default runtime, prompts, and RAG configuration.

## 5. Start LlamaFarm

```bash
lf start
```

This command:
- Starts the API server and Universal Runtime natively
- Opens the interactive chat TUI
- Launches the Designer web UI at `http://localhost:8000`

Hit `Ctrl+C` to exit the chat UI when you're done.

:::tip Use the Designer Web UI
Prefer a visual interface? Open `http://localhost:8000` in your browser to access the Designer—manage projects, upload datasets, configure models, and test prompts without touching the command line.

See the [Designer documentation](../designer/index.md) for details.
:::

### Running Services Manually

For development, you can run services individually:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Start both services
nx dev

# Or in separate terminals:
nx start server           # Terminal 1
nx start universal-runtime # Terminal 2
```

## 6. Chat with Your Project

```bash
# Interactive chat (opens TUI)
lf chat

# One-off message
lf chat "What can you do?"
```

Useful options:
- `--no-rag` — Bypass retrieval, hit the model directly
- `--database`, `--retrieval-strategy` — Override RAG behavior
- `--curl` — Print the equivalent curl command

## 7. Create and Populate a Dataset

```bash
# Create a dataset
lf datasets create -s pdf_ingest -b main_db research-notes

# Upload documents (supports globs/directories)
lf datasets upload research-notes ./examples/fda_rag/files/*.pdf
```

## 8. Process Documents

```bash
lf datasets process research-notes
```

This sends documents through the RAG pipeline—parsing, chunking, embedding, and indexing.

For large PDFs, processing may take a few minutes. The CLI shows progress indicators.

## 9. Query with RAG

```bash
lf rag query --database main_db "What are the key findings?"
```

Useful flags:
- `--top-k 10` — Number of results
- `--filter "file_type:pdf"` — Metadata filtering
- `--include-metadata` — Show document sources
- `--include-score` — Show relevance scores

## 10. Next Steps

- [Designer Web UI](../designer/index.md) — Visual interface for managing projects
- [Configuration Guide](../configuration/index.md) — Deep dive into `llamafarm.yaml`
- [RAG Guide](../rag/index.md) — Strategies, parsers, and retrieval
- [ML Models](../models/specialized-ml.md) — Classifiers, OCR, anomaly detection
- [API Reference](../api/index.md) — Full REST API documentation
- [Examples](../examples/index.md) — Run the FDA and Raleigh demos end-to-end

Need help? Chat with us on [Discord](https://discord.gg/RrAUXTCVNF) or open a [discussion](https://github.com/llama-farm/llamafarm/discussions).
