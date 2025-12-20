---
sidebar_position: 1
sidebar_label: Start Here
---

# Welcome to LlamaFarm

LlamaFarm helps you ship retrieval-augmented and agentic AI apps from your laptop to production. It is fully open-source and intentionally extendable—swap model providers, vector stores, parsers, and CLI workflows without rewriting your project.

:::info Found a bug or have a feature request?
[Submit an issue on GitHub →](https://github.com/llama-farm/llamafarm/issues)
:::

## 🖥️ Desktop App

Get started instantly with our desktop application — no command line required.

<div style={{display: 'flex', gap: '16px', flexWrap: 'wrap', marginBottom: '24px', marginTop: '16px'}}>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-mac-arm64.dmg" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Mac (M1+)
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-mac-universal.dmg" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Mac (Intel)
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-windows.exe" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Windows
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-x86_64.AppImage" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Linux (x86_64)
  </a>
  <a href="https://github.com/llama-farm/llamafarm/releases/latest/download/LlamaFarm-desktop-app-linux-arm64.AppImage" style={{display: 'inline-flex', alignItems: 'center', padding: '12px 24px', backgroundColor: '#2563eb', color: 'white', borderRadius: '8px', textDecoration: 'none', fontWeight: '600', fontSize: '16px'}}>
    ⬇️ Linux (ARM64)
  </a>
</div>

The desktop app includes everything you need: visual project management, dataset uploads, chat interface, and built-in model management. [**See hardware requirements →**](./desktop-app/index.md)

## 📺 Video Demo

**Quick Overview (90 seconds):** https://youtu.be/W7MHGyN0MdQ

Get a fast introduction to LlamaFarm's core features and see it in action.

## What You Can Do Today

- **Prototype locally** with Ollama or any OpenAI-compatible runtime (vLLM, Together, custom gateways).
- **Ingest and query documents** using configurable RAG pipelines defined entirely in YAML.
- **Choose your interface** – Use the powerful `lf` CLI for automation and scripting, or the Designer web UI for visual project management with drag-and-drop dataset uploads and interactive configuration.
- **Extend everything** from model handlers to data processors by updating schemas and wiring your own implementations.
- **Give models superpowers** with MCP (Model Context Protocol) – Connect your AI to local tools, APIs, and databases through a standardized protocol.

## MCP (Model Context Protocol)

LlamaFarm supports **MCP** – a standardized protocol for giving AI models access to external tools. Connect your AI to filesystems, databases, APIs, and custom business logic.

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

**Key features:**
- **Per-model access control** – Different models get different tools
- **Multiple transports** – STDIO (local), HTTP (remote), SSE (streaming)
- **Persistent sessions** – Efficient connection management

[**Learn more about MCP →**](./mcp/index.md)

## Choose Your Own Adventure

| Get Started                                                                           | Go Deeper                                                                         | Build Your Own                                                                                 |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [Quickstart](./quickstart/index.md) – install, init, chat, ingest your first dataset. | [Core Concepts](./concepts/index.md) – architecture, sessions, and components.    | [Extending LlamaFarm](./extending/index.md) – add runtimes, stores, parsers, and CLI commands. |
| [Designer Web UI](./designer/index.md) – visual interface for project management.     | [Configuration Guide](./configuration/index.md) – schema-driven project settings. | [RAG Guide](./rag/index.md) – strategies, processing pipelines, and monitoring.                |
| [CLI Reference](./cli/index.md) – command matrix and examples.                        | [Models & Runtime](./models/index.md) – configure AI models and providers.        | [Prompts](./prompts/index.md) – prompt engineering and management.                             |

## Philosophy

- **Local-first, cloud-aware** – everything works offline, yet you can point at remote runtimes when needed.
- **Configuration over code** – projects are reproducible because behaviour lives in `llamafarm.yaml`.
- **Composable modules** – RAG, prompts, and runtime selection work independently but integrate cleanly.
- **Flexible interfaces** – Use the CLI for automation, the Designer for visual management, or the REST API for custom integrations.
- **Open for extension** – documentation includes patterns for registering new providers, stores, and utilities.

:::tip Prefer Visual Tools?
The **Designer Web UI** provides a browser-based interface for managing projects, uploading datasets, and testing your AI—all without touching the command line. It's automatically available at `http://localhost:8000` when you run `lf start`. [Learn more →](./designer/index.md)
:::

## 🎥 In-Depth Tutorial

**Complete Walkthrough (7 minutes):** https://youtu.be/HNnZ4iaOSJ4

Watch a comprehensive demonstration of LlamaFarm's features including project setup, dataset ingestion, RAG queries, and configuration options.

---

Ready to build? Start with the [Quickstart](./quickstart/index.md) and keep the CLI open in another terminal.
