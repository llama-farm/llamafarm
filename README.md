# LlamaFarm - Edge AI for Everyone

> Enterprise AI capabilities on your own hardware. No cloud required.

[![License: Apache 2.0](https://img.shields.io/github/license/llama-farm/llamafarm)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Go 1.24+](https://img.shields.io/badge/go-1.24+-00ADD8.svg)](https://go.dev/dl/)
[![Docs](https://img.shields.io/badge/docs-latest-4C51BF.svg)](https://docs.llamafarm.dev)
[![Discord](https://img.shields.io/discord/1392890421771899026.svg)](https://discord.gg/RrAUXTCVNF)

**LlamaFarm** is an open-source AI platform that runs entirely on your hardware. Build RAG applications, train custom classifiers, detect anomalies, and run document processing—all locally with complete privacy.

- 🔒 **Complete Privacy** — Your data never leaves your device
- 💰 **No API Costs** — Use open-source models without per-token fees
- 🌐 **Offline Capable** — Works without internet once models are downloaded
- ⚡ **Hardware Optimized** — Automatic GPU/NPU acceleration on Apple Silicon, NVIDIA, and AMD

### Desktop App Downloads

Get started instantly — no command line required:

| Platform | Download |
|----------|----------|
| **Mac (M1+)** | [Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-mac-arm64.dmg) |
| **Mac (Intel/Universal)** | [Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-mac-universal.dmg) |
| **Windows** | [Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-windows.exe) |
| **Linux (x86_64)** | [Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-x86_64.AppImage) |
| **Linux (ARM64)** | [Download](https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-arm64.AppImage) |

---

### What Can You Build?

| Capability | Description |
|-----------|-------------|
| **RAG (Retrieval-Augmented Generation)** | Ingest PDFs, docs, CSVs and query them with AI |
| **Custom Classifiers** | Train text classifiers with 8-16 examples using SetFit |
| **Anomaly Detection** | Detect outliers in logs, metrics, or transactions |
| **OCR & Document Extraction** | Extract text and structured data from images and PDFs |
| **Named Entity Recognition** | Find people, organizations, and locations |
| **Multi-Model Runtime** | Switch between Ollama, OpenAI, vLLM, or local GGUF models |

**Video demo (90 seconds):** https://youtu.be/W7MHGyN0MdQ

---

## Quickstart

### Option 1: Desktop App

Download the desktop app above and run it. No additional setup required.

### Option 2: CLI + Development Mode

1. **Install the CLI**

   macOS / Linux:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```

   Windows (PowerShell):
   ```powershell
   irm https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.ps1 | iex
   ```

   Or download directly from [releases](https://github.com/llama-farm/llamafarm/releases/latest).

2. **Create and run a project**

   ```bash
   lf init my-project      # Generates llamafarm.yaml
   lf start                # Starts services and opens Designer UI
   ```

3. **Chat with your AI**

   ```bash
   lf chat                           # Interactive chat
   lf chat "Hello, LlamaFarm!"       # One-off message
   ```

The Designer web interface is available at `http://localhost:8000`.

### Option 3: Development from Source

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

# Install Nx globally and initialize the workspace
npm install -g nx
nx init --useDotNxInstallation --interactive=false  # Required on first clone

# Start all services (run each in a separate terminal)
nx start server           # FastAPI server (port 8000)
nx start rag              # RAG worker for document processing
nx start universal-runtime # ML models, OCR, embeddings (port 11540)
```

---

## Architecture

LlamaFarm consists of three main services:

| Service | Port | Purpose |
|---------|------|---------|
| **Server** | 8000 | FastAPI REST API, Designer web UI, project management |
| **RAG Worker** | - | Celery worker for async document processing |
| **Universal Runtime** | 11540 | ML model inference, embeddings, OCR, anomaly detection |

All configuration lives in `llamafarm.yaml`—no scattered settings or hidden defaults.

---

## Runtime Options

### Universal Runtime (Recommended)

The Universal Runtime provides access to HuggingFace models plus specialized ML capabilities:

- **Text Generation** - Any HuggingFace text model
- **Embeddings** - sentence-transformers and other embedding models
- **OCR** - Text extraction from images/PDFs (Surya, EasyOCR, PaddleOCR, Tesseract)
- **Document Extraction** - Forms, invoices, receipts via vision models
- **Text Classification** - Pre-trained or custom models via SetFit
- **Named Entity Recognition** - Extract people, organizations, locations
- **Reranking** - Cross-encoder models for improved RAG quality
- **Anomaly Detection** - Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoders

```yaml
runtime:
  models:
    default:
      provider: universal
      model: Qwen/Qwen2.5-1.5B-Instruct
      base_url: http://127.0.0.1:11540/v1
```

### Ollama

Simple setup for GGUF models with CPU/GPU acceleration:

```yaml
runtime:
  models:
    default:
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434/v1
```

### OpenAI-Compatible

Works with vLLM, Together, Mistral API, or any OpenAI-compatible endpoint:

```yaml
runtime:
  models:
    default:
      provider: openai
      model: gpt-4o
      base_url: https://api.openai.com/v1
      api_key: ${OPENAI_API_KEY}
```

---

## Core Workflows

### CLI Commands

| Task | Command |
|------|---------|
| Initialize project | `lf init my-project` |
| Start services | `lf start` |
| Interactive chat | `lf chat` |
| One-off message | `lf chat "Your question"` |
| List models | `lf models list` |
| Use specific model | `lf chat --model powerful "Question"` |
| Create dataset | `lf datasets create -s pdf_ingest -b main_db research` |
| Upload files | `lf datasets upload research ./docs/*.pdf` |
| Process dataset | `lf datasets process research` |
| Query RAG | `lf rag query --database main_db "Your query"` |
| Check RAG health | `lf rag health` |

### RAG Pipeline

1. **Create a dataset** linked to a processing strategy and database
2. **Upload files** (PDF, DOCX, Markdown, TXT)
3. **Process** to parse, chunk, and embed documents
4. **Query** using semantic search with optional metadata filtering

```bash
lf datasets create -s default -b main_db research
lf datasets upload research ./papers/*.pdf
lf datasets process research
lf rag query --database main_db "What are the key findings?"
```

### Designer Web UI

The Designer at `http://localhost:8000` provides:

- Visual dataset management with drag-and-drop uploads
- Interactive configuration editor with live validation
- Integrated chat with RAG context
- Switch between visual and YAML editing modes

---

## Configuration

`llamafarm.yaml` is the source of truth for each project:

```yaml
version: v1
name: my-assistant
namespace: default

# Multi-model configuration
runtime:
  default_model: fast

  models:
    fast:
      description: "Fast local model"
      provider: universal
      model: Qwen/Qwen2.5-1.5B-Instruct
      base_url: http://127.0.0.1:11540/v1

    powerful:
      description: "More capable model"
      provider: universal
      model: Qwen/Qwen2.5-7B-Instruct
      base_url: http://127.0.0.1:11540/v1

# System prompts
prompts:
  - name: default
    messages:
      - role: system
        content: You are a helpful assistant.

# RAG configuration
rag:
  databases:
    - name: main_db
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: semantic_search
      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          config:
            model: sentence-transformers/all-MiniLM-L6-v2
            base_url: http://127.0.0.1:11540/v1
      retrieval_strategies:
        - name: semantic_search
          type: BasicSimilarityStrategy
          config:
            top_k: 5

  data_processing_strategies:
    - name: default
      parsers:
        - type: PDFParser_LlamaIndex
          config:
            chunk_size: 1000
            chunk_overlap: 100
        - type: MarkdownParser_Python
          config:
            chunk_size: 1000
      extractors: []

# Dataset definitions
datasets:
  - name: research
    data_processing_strategy: default
    database: main_db
```

### Environment Variable Substitution

Use `${VAR}` syntax to inject secrets from `.env` files:

```yaml
runtime:
  models:
    openai:
      api_key: ${OPENAI_API_KEY}
      # With default: ${OPENAI_API_KEY:-sk-default}
      # From specific file: ${file:.env.production:API_KEY}
```

See the [Configuration Guide](docs/website/docs/configuration/index.md) for complete reference.

---

## REST API

LlamaFarm provides an OpenAI-compatible REST API:

**Chat Completions**
```bash
curl -X POST http://localhost:8000/v1/projects/default/my-project/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "rag_enabled": true
  }'
```

**RAG Query**
```bash
curl -X POST http://localhost:8000/v1/projects/default/my-project/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements?",
    "database": "main_db",
    "top_k": 5
  }'
```

See the [API Reference](docs/website/docs/api/index.md) for all endpoints.

---

## Specialized ML Capabilities

The Universal Runtime provides endpoints beyond chat:

### OCR & Document Extraction

```bash
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F "file=@document.pdf" \
  -F "model=surya"
```

### Anomaly Detection

```bash
# Train on normal data
curl -X POST http://localhost:8000/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{"model": "sensor-detector", "backend": "isolation_forest", "data": [[22.1], [23.5], ...]}'

# Detect anomalies
curl -X POST http://localhost:8000/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"model": "sensor-detector", "data": [[22.0], [100.0], [23.0]], "threshold": 0.5}'
```

### Text Classification & NER

See the [Models Guide](docs/website/docs/models/index.md) for complete documentation.

---

## Examples

| Example | Description | Location |
|---------|-------------|----------|
| FDA Letters Assistant | Multi-PDF ingestion, regulatory queries | `examples/fda_rag/` |
| Raleigh Planning Helper | Large ordinance documents, geospatial queries | `examples/gov_rag/` |
| OCR & Document Processing | Image text extraction, form parsing | `examples/ocr_and_document/` |

---

## Development & Testing

```bash
# Python server tests
cd server && uv sync && uv run --group test python -m pytest

# CLI tests
cd cli && go test ./...

# RAG tests
cd rag && uv sync && uv run pytest tests/

# Universal Runtime tests
cd runtimes/universal && uv sync && uv run pytest tests/

# Build docs
nx build docs
```

---

## Extensibility

- **Add runtimes** by implementing provider support and updating schema
- **Add vector stores** by implementing store backends (Chroma, Qdrant, etc.)
- **Add parsers** for new file formats (PDF, DOCX, HTML, CSV, etc.)
- **Add extractors** for custom metadata extraction
- **Add CLI commands** under `cli/cmd/`

See the [Extending Guide](docs/website/docs/extending/index.md) for step-by-step instructions.

---

## Community & Support

- [Discord](https://discord.gg/RrAUXTCVNF) - Chat with the team and community
- [GitHub Issues](https://github.com/llama-farm/llamafarm/issues) - Bug reports and feature requests
- [Discussions](https://github.com/llama-farm/llamafarm/discussions) - Ideas and proposals
- [Contributing Guide](CONTRIBUTING.md) - Code style and contribution process

---

## License

Licensed under the [Apache 2.0 License](LICENSE). See [CREDITS](CREDITS.md) for acknowledgments.

---

Build locally. Deploy anywhere. Own your AI.
