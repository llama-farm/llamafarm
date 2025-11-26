# 🦙 LlamaFarm - Run your own AI anywhere

> Build powerful AI locally, extend anywhere.

[![License: Apache 2.0](https://img.shields.io/github/license/llama-farm/llamafarm)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Go 1.24+](https://img.shields.io/badge/go-1.24+-00ADD8.svg)](https://go.dev/dl/)
[![Docs](https://img.shields.io/badge/docs-latest-4C51BF.svg)](docs/website/docs/intro.md)
[![Discord](https://img.shields.io/discord/1392890421771899026.svg)](https://discord.gg/RrAUXTCVNF)

LlamaFarm is an open-source framework for building retrieval-augmented and agentic AI applications. It ships with opinionated defaults (Ollama for local models, Chroma for vector storage) while staying 100% extendable—swap in vLLM, remote OpenAI-compatible hosts, new parsers, or custom stores without rewriting your app.

- **Local-first developer experience** with a single CLI (`lf`) that manages projects, datasets, and chat sessions.
- **Production-ready architecture** that mirrors server endpoints and enforces schema-based configuration.
- **Composable RAG pipelines** you can tailor through YAML, not bespoke code.
- **Extendable everything**: runtimes, embedders, databases, extractors, and CLI tooling.

**📺 Video demo (90 seconds):** https://youtu.be/W7MHGyN0MdQ

---

## 🚀 Quickstart (TL;DR)

**Prerequisites:**

- [Docker](https://www.docker.com/get-started/)
- [Ollama](https://ollama.com/download) *(local runtime)* **OR** [Lemonade](runtimes/lemonade/QUICKSTART.md) *(local GGUF models with NPU/GPU acceleration)*

1. **Install the CLI**

   macOS / Linux
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```

   Windows (via winget)
   ```
   winget install LlamaFarm.CLI
   ```

2. **Adjust Ollama context window**
   - Open the Ollama app, go to **Settings → Advanced**, and set the context window to match production (e.g., 100K tokens).
   - Larger context windows improve RAG answers when long documents are ingested.

3. **Create and run a project**
   ```bash
   lf init my-project            # Generates llamafarm.yaml using the server template
   lf start                      # Spins up Docker services, opens dev chat UI & Designer web UI
   ```
   
   The Designer web interface will be available at `http://localhost:7724` for visual project management.

4. **Start an interactive project chat or send a one-off message**
```bash
# Interactive project chat (auto-detects namespace/project from llamafarm.yaml)
lf chat

# One-off message
lf chat "Hello, LlamaFarm!"
```

Need the full walkthrough with dataset ingestion and troubleshooting tips? Jump to the [Quickstart guide](docs/website/docs/quickstart/index.md).

> Prefer building from source? Clone the repo and follow the steps in [Development & Testing](#-development--testing).

**Run services manually (without Docker auto-start):**

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

# Install Nx globally and bootstrap the workspace
npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Option 1: start both server and RAG worker with one command
nx dev

# Option 2: start services in separate terminals
# Terminal 1
nx start rag
# Terminal 2
nx start server
```

Open another terminal to run `lf` commands (installed or built from source). This is equivalent to what `lf start` orchestrates automatically.

---

## 🌟 Why LlamaFarm

- **Own your stack** – Run small local models today and swap to hosted vLLM, Together, or custom APIs tomorrow by changing `llamafarm.yaml`.
- **Battle-tested RAG** – Configure parsers, extractors, embedding strategies, and databases without touching orchestration code.
- **Config over code** – Every project is defined by YAML schemas that are validated at runtime and easy to version control.
- **Friendly CLI & Web UI** – `lf` CLI handles project bootstrapping, dataset lifecycle, and RAG queries. Prefer visual tools? Use the Designer web interface for drag-and-drop dataset management and interactive configuration.
- **Built to extend** – Add a new provider or vector store by registering a backend and regenerating schema types.

---

## 🔧 Core Workflows

### CLI Commands

| Task | Command | Notes |
| ---- | ------- | ----- |
| Initialize a project | `lf init my-project` | Creates `llamafarm.yaml` from server template. |
| Start dev stack + chat TUI | `lf start` | Spins up server, rag worker, monitors Ollama/vLLM. |
| Interactive project chat | `lf chat` | Opens TUI using project from `llamafarm.yaml`. |
| List available models | `lf models list` | Shows all configured models (Ollama, Lemonade, OpenAI, etc.). |
| Use specific model | `lf chat --model powerful "question"` | Switch between models in multi-model configs. |
| Send single prompt | `lf chat "Explain retrieval augmented generation"` | Uses RAG by default; add `--no-rag` for pure LLM. |
| Preview REST call | `lf chat --curl "What models are configured?"` | Prints sanitized `curl` command. |
| Create dataset | `lf datasets create -s pdf_ingest -b main_db research-notes` | Validates strategy/database against project config. |
| Upload files | `lf datasets upload research-notes ./docs/*.pdf` | Supports globs and directories. |
| Process dataset | `lf datasets process research-notes` | Streams heartbeat dots during long processing. |
| Semantic query | `lf rag query --database main_db "What did the 2024 FDA letters require?"` | Use `--filter`, `--include-metadata`, etc. |

See the [CLI reference](docs/website/docs/cli/index.md) for full command details and troubleshooting advice.

### Designer Web UI

Prefer a visual interface? The Designer provides a browser-based way to manage projects:

- **Visual dataset management** – Drag-and-drop file uploads, see processing status in real-time
- **Interactive configuration** – Edit your project settings with live validation and helpful hints
- **Integrated chat** – Test your AI with full RAG context directly in the browser
- **Dual-mode editing** – Switch between visual designer and direct YAML editing

Access the Designer at `http://localhost:7724` after running `lf start`, or at `http://localhost:3123` when using Docker Compose.

See the [Designer documentation](docs/website/docs/designer/index.md) for a complete feature walkthrough.

---

## 🌐 REST API

LlamaFarm provides a comprehensive REST API (compatible with OpenAI's format) for integrating with your applications. The API runs at `http://localhost:8000`.

### Key Endpoints

**Chat Completions** (OpenAI-compatible)
```bash
curl -X POST http://localhost:8000/v1/projects/{namespace}/{project}/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What are the FDA requirements?"}
    ],
    "stream": false,
    "rag_enabled": true,
    "database": "main_db"
  }'
```

**RAG Query**
```bash
curl -X POST http://localhost:8000/v1/projects/{namespace}/{project}/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "clinical trial requirements",
    "database": "main_db",
    "top_k": 5
  }'
```

**Dataset Management**
```bash
# Upload file
curl -X POST http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/data \
  -F "file=@document.pdf"

# Start async processing (returns Celery task info)
curl -X POST http://localhost:8000/v1/projects/{namespace}/{project}/datasets/{dataset}/actions \
  -H "Content-Type: application/json" \
  -d '{"action_type":"process"}'
```

### Finding Your Namespace and Project

Check your `llamafarm.yaml`:
```yaml
name: my-project        # Your project name
namespace: my-org       # Your namespace
```

Or inspect the file system: `~/.llamafarm/projects/{namespace}/{project}/`

See the complete [API Reference](docs/website/docs/api/index.md) for all endpoints, request/response formats, Python/TypeScript clients, and examples.

---

## 🗂️ Configuration Snapshot

`llamafarm.yaml` is the source of truth for each project. The schema enforces required fields and documents every extension point.

### Multi-Model Configuration (Recommended)

```yaml
version: v1
name: fda-assistant
namespace: default

runtime:
  default_model: fast                # Which model to use by default

  models:
    fast:
      description: "Fast Ollama model"
      provider: ollama
      model: gemma3:1b

    powerful:
      description: "More capable Ollama model"
      provider: ollama
      model: qwen3:8b

    lemon:
      description: "Lemonade local model with NPU/GPU"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are an FDA specialist. Answer using short paragraphs and cite document titles when available.

rag:
  databases:
    - name: main_db
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: filtered_search
      embedding_strategies:
        - name: default_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text:latest
      retrieval_strategies:
        - name: filtered_search
          type: MetadataFilteredStrategy
          config:
            top_k: 5
  data_processing_strategies:
    - name: pdf_ingest
      parsers:
        - type: PDFParser_LlamaIndex
          config:
            chunk_size: 1500
            chunk_overlap: 200
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor

datasets:
  - name: research-notes
    data_processing_strategy: pdf_ingest
    database: main_db
```

**Using your models:**
```bash
lf models list                              # See all configured models
lf chat "Question"                          # Uses default model (fast)
lf chat --model powerful "Complex question" # Use specific model
lf chat --model lemon "Local GGUF model"    # Use Lemonade model
```

> **Note:** Lemonade models require manual startup via `nx start lemonade` from the project root. The `nx start lemonade` command automatically picks up configuration from your `llamafarm.yaml`. In the future, Lemonade will run as a container and be auto-started. See [Lemonade Quickstart](runtimes/lemonade/QUICKSTART.md) for setup.

Configuration reference: [Configuration Guide](docs/website/docs/configuration/index.md) • [Extending LlamaFarm](docs/website/docs/extending/index.md)

---

## 🧩 Extensibility Highlights

- **Swap runtimes** by pointing to any OpenAI-compatible endpoint (vLLM, Mistral, Anyscale). Update `runtime.provider`, `base_url`, and `api_key`; regenerate schema types if you add a new provider enum.
- **Bring your own vector store** by implementing a store backend, adding it to `rag/schema.yaml`, and updating the server service registry.
- **Add parsers/extractors** to support new file formats or metadata pipelines. Register implementations and extend the schema definitions.
- **Extend the CLI** with new Cobra commands under `cli/cmd`; the docs include guidance on adding dataset utilities or project tooling.

Check the [Extending guide](docs/website/docs/extending/index.md) for step-by-step instructions.

---

## 📚 Examples

| Example | What it Shows | Location |
| ------- | ------------- | -------- |
| FDA Letters Assistant | Multi-document PDF ingestion, RAG queries, reference-style prompts | `examples/fda_rag/` & [Docs](docs/website/docs/examples/index.md#fda-letters-assistant) |
| Raleigh UDO Planning Helper | Large ordinance ingestion, long-running processing tips, geospatial queries | `examples/gov_rag/` & [Docs](docs/website/docs/examples/index.md#raleigh-udo-planning-helper) |

Run `lf datasets` and `lf rag query` commands from each example folder to reproduce the flows demonstrated in the docs.

---

## 🧪 Development & Testing

```bash
# Python server + RAG tests
cd server
uv sync
uv run --group test python -m pytest

# CLI tests
cd ../cli
go test ./...

# RAG tooling smoke tests
cd ../rag
uv sync
uv run python cli.py test

# Docs build (ensures navigation/link integrity)
cd ..
nx build docs
```

Linting: `uv run ruff check --fix .` (Python), `go fmt ./...` and `go vet ./...` (Go).

---

## 🤝 Community & Support

- [Discord](https://discord.gg/RrAUXTCVNF) – chat with the team, share feedback, find collaborators.
- [GitHub Issues](https://github.com/llama-farm/llamafarm/issues) – bug reports and feature requests.
- [Discussions](https://github.com/llama-farm/llamafarm/discussions) – ideas, RFCs, roadmap proposals.
- [Contributing Guide](CONTRIBUTING.md) – code style, testing expectations, doc updates, schema regeneration steps.

Want to add a new provider, parser, or example? Start a discussion or open a draft PR—we love extensions!

---

## 📄 License & Acknowledgments

- Licensed under the [Apache 2.0 License](LICENSE).
- Built by the LlamaFarm community and inspired by the broader open-source AI ecosystem. See [CREDITS](CREDITS.md) for detailed acknowledgments.

---

Build locally. Deploy anywhere. Own your AI.
