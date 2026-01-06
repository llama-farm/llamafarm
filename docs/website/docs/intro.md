---
sidebar_position: 1
sidebar_label: Start Here
---

# Welcome to LlamaFarm

**LlamaFarm brings enterprise AI capabilities to everyone.** Run powerful language models, document processing, and intelligent retrieval—all locally on your hardware. No cloud required. No data leaves your machine.

:::info Found a bug or have a feature request?
[Submit an issue on GitHub →](https://github.com/llama-farm/llamafarm/issues)
:::

## Why LlamaFarm?

### 🔒 Edge AI for Everyone

Run sophisticated AI workloads on your own hardware:

- **Complete Privacy** — Your documents, queries, and data never leave your device
- **No API Costs** — Use open-source models without per-token fees
- **Offline Capable** — Works without internet once models are downloaded
- **Hardware Optimized** — Automatic GPU/NPU acceleration on Apple Silicon, NVIDIA, and AMD

### 🧠 Production-Ready AI Stack

LlamaFarm isn't just a wrapper—it's a complete AI development platform:

| Capability | What It Does |
|-----------|--------------|
| **RAG (Retrieval-Augmented Generation)** | Ingest PDFs, docs, CSVs and query them with AI. Your documents become searchable knowledge. |
| **Multi-Model Runtime** | Switch between Ollama, OpenAI, vLLM, or local GGUF models in one config file. |
| **Custom Classifiers** | Train text classifiers with 8-16 examples using SetFit. No ML expertise required. |
| **Anomaly Detection** | Detect outliers in logs, metrics, or transactions with one API call. |
| **OCR & Document Extraction** | Extract text and structured data from images and PDFs. |
| **Named Entity Recognition** | Find people, organizations, and locations in your text. |
| **Agentic Tools (MCP)** | Give AI models access to filesystems, databases, and APIs. |

### ⚡ Developer Experience

- **Config-Driven** — Define your entire AI stack in `llamafarm.yaml`
- **CLI + Web UI** — Use the `lf` command line or the Designer visual interface
- **REST API** — OpenAI-compatible endpoints for easy integration
- **Extensible** — Add custom parsers, embedders, and model providers

---

## Get Started in 60 Seconds

### Option 1: Desktop App (Easiest)

Download the all-in-one desktop application:

<div style={{display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '24px', marginTop: '16px'}}>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Mac (M1+)
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Windows
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Linux
  </a>
</div>

The desktop app bundles everything: server, Universal Runtime, and the Designer web UI.

### Option 2: CLI Installation

Install the `lf` command-line tool:

**macOS / Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.ps1 | iex
```

**Or download directly:**
- [Latest Release](https://github.com/llama-farm/llamafarm/releases/latest) — Download `lf` binary for your platform

Verify installation:
```bash
lf --help
```

---

## 📺 See It In Action

**Quick Overview (90 seconds):** https://youtu.be/W7MHGyN0MdQ

**Complete Walkthrough (7 minutes):** https://youtu.be/HNnZ4iaOSJ4

---

## What Can You Build?

### Document Q&A
Upload your company's documents and ask questions in natural language:
```bash
lf datasets upload knowledge-base ./contracts/*.pdf
lf datasets process knowledge-base
lf chat "What are our standard payment terms?"
```

### Custom Intent Classification
Train a classifier to route support tickets:
```python
# Train with just 8 examples per category
POST /v1/ml/classifier/fit
{
  "model": "ticket-router",
  "training_data": [
    {"text": "I can't log in", "label": "auth"},
    {"text": "Charge me twice", "label": "billing"},
    ...
  ]
}
```

### Real-Time Anomaly Detection
Monitor API logs for suspicious activity:
```python
# Train on normal traffic
POST /v1/ml/anomaly/fit
{"model": "api-monitor", "data": [...normal_requests...]}

# Detect anomalies in real-time
POST /v1/ml/anomaly/detect
{"model": "api-monitor", "data": [...new_requests...]}
```

### Document Processing Pipeline
Extract structured data from invoices and forms:
```bash
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F "file=@invoice.pdf" \
  -F "model=surya"
```

---

## Choose Your Path

| Get Started | Go Deeper | Build Your Own |
|-------------|-----------|----------------|
| [Quickstart](./quickstart/index.md) — Install, init, chat, ingest your first dataset | [Core Concepts](./concepts/index.md) — Architecture, sessions, and components | [Extending LlamaFarm](./extending/index.md) — Add runtimes, stores, parsers |
| [Designer Web UI](./designer/index.md) — Visual interface for project management | [Configuration Guide](./configuration/index.md) — Schema-driven project settings | [RAG Guide](./rag/index.md) — Strategies, processing pipelines |
| [CLI Reference](./cli/index.md) — Command matrix and examples | [Models & Runtime](./models/index.md) — Configure AI models and providers | [API Reference](./api/index.md) — Full REST API documentation |

---

## Philosophy

- **Local-first, cloud-aware** — Everything works offline, yet you can point at remote runtimes when needed
- **Configuration over code** — Projects are reproducible because behavior lives in `llamafarm.yaml`
- **Composable modules** — RAG, prompts, and runtime selection work independently but integrate cleanly
- **Edge for everyone** — Enterprise AI capabilities without enterprise infrastructure
- **Open for extension** — Add custom providers, stores, and utilities

---

## Advanced: MCP (Model Context Protocol)

LlamaFarm supports **MCP** for giving AI models access to external tools like filesystems, databases, and APIs.

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir']

runtime:
  models:
    - name: assistant
      provider: openai
      model: gpt-4
      mcp_servers: [filesystem]
```

[**Learn more about MCP →**](./mcp/index.md)

---

Ready to build? Start with the [Quickstart](./quickstart/index.md).
