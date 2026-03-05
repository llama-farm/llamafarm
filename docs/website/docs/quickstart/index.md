---
title: Quickstart
sidebar_position: 1
---

# Quickstart

Get LlamaFarm installed, ingest a dataset, and run your first RAG-powered chat in minutes.

## 1. Install LlamaFarm

### Option A: Desktop App (Easiest)

Download the all-in-one desktop application:

| Platform            | Download                                                                                                                    |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Mac (Universal)** | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-mac-universal.dmg)     |
| **Windows**         | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-windows.exe)           |
| **Linux (x86_64)**  | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-x86_64.AppImage) |
| **Linux (arm64)**   | [⬇️ Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-arm64.AppImage)  |

The desktop app bundles everything you need—no additional installation required. Launch it and the built-in Designer will guide you through project setup with an onboarding wizard.

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

| Platform              | Binary                 |
| --------------------- | ---------------------- |
| macOS (Apple Silicon) | `lf-darwin-arm64`      |
| macOS (Intel)         | `lf-darwin-amd64`      |
| Linux (x64)           | `lf-linux-amd64`       |
| Linux (arm64)         | `lf-linux-arm64`       |
| Windows (x86_64)      | `lf-windows-amd64.exe` |

After downloading, make it executable and add to your PATH:

```bash
chmod +x lf-darwin-arm64
sudo mv lf-darwin-arm64 /usr/local/bin/lf
```

Verify installation:

```bash
lf --help
```

## 2. Create a Project

```bash
lf init my-project
cd my-project
```

This creates `llamafarm.yaml` with default runtime, prompts, and RAG configuration.

## 3. Start LlamaFarm

```bash
lf start
```

This command:

- Starts the API server and Universal Runtime natively
- Opens the interactive chat TUI
- Launches the Designer web UI at `http://localhost:14345`

Hit `Ctrl+C` to exit the chat UI when you're done.

### Use the Designer Web UI

Open `http://localhost:14345` in your browser to access the **Designer** — a visual interface for managing your entire project without touching the command line.

When you open a new project in the Designer, the **onboarding wizard** walks you through setup step by step:

1. **What are you building?** — Describe your project's purpose
2. **Pick a model** — Choose from available local models or connect an API
3. **Add data** — Upload documents for RAG or skip for later
4. **Test it out** — Send your first message and see it work

The wizard generates your `llamafarm.yaml` configuration automatically. After setup, you get full access to the [Dashboard](../designer/dashboard.md), [Prompts editor](../designer/prompts.md), [Data management](../designer/data-management.md), [RAG configuration](../designer/databases.md), [Model settings](../designer/models.md), and [Test interface](../designer/chat-and-testing.md).

See the [Designer documentation](../designer/index.md) for the full guide.

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

## 4. Chat with Your Project

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

## 5. Create and Populate a Dataset

```bash
# Create a dataset
lf datasets create -s pdf_ingest -b main_db research-notes

# Upload documents (supports globs/directories); auto-processes by default
lf datasets upload research-notes ./examples/fda_rag/files/*.pdf
# For batching without processing:
# lf datasets upload research-notes ./examples/fda_rag/files/*.pdf --no-process
```

## 6. Process Documents

```bash
lf datasets process research-notes    # Only needed if you skipped auto-processing
```

This sends documents through the RAG pipeline—parsing, chunking, embedding, and indexing.

For large PDFs, processing may take a few minutes. The CLI shows progress indicators.

## 7. Query with RAG

```bash
lf rag query --database main_db "What are the key findings?"
```

Useful flags:

- `--top-k 10` — Number of results
- `--filter "file_type:pdf"` — Metadata filtering
- `--include-metadata` — Show document sources
- `--include-score` — Show relevance scores

## 8. Next Steps

- [Designer Web UI](../designer/index.md) — Visual interface for managing projects
- [Configuration Guide](../configuration/index.md) — Deep dive into `llamafarm.yaml`
- [RAG Guide](../rag/index.md) — Strategies, parsers, and retrieval
- [ML Models](../models/specialized-ml.md) — Classifiers, OCR, anomaly detection
- [API Reference](../api/index.md) — Full REST API documentation
- [Examples](../examples/index.md) — Run the FDA and Raleigh demos end-to-end

Need help? Chat with us on [Discord](https://discord.gg/RrAUXTCVNF) or open a [discussion](https://github.com/llama-farm/llamafarm/discussions).
