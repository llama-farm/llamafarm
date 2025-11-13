---
sidebar_position: 1
sidebar_label: Start Here
---

# Welcome to LlamaFarm

LlamaFarm helps you ship retrieval-augmented and agentic AI apps from your laptop to production. It is fully open-source and intentionally extendable—swap model providers, vector stores, parsers, and CLI workflows without rewriting your project.

## 📺 Video Demo

**Quick Overview (90 seconds):** https://youtu.be/W7MHGyN0MdQ

Get a fast introduction to LlamaFarm's core features and see it in action.

## What You Can Do Today

- **Prototype locally** with Ollama or any OpenAI-compatible runtime (vLLM, Together, custom gateways).
- **Ingest and query documents** using configurable RAG pipelines defined entirely in YAML.
- **Choose your interface** – Use the powerful `lf` CLI for automation and scripting, or the Designer web UI for visual project management with drag-and-drop dataset uploads and interactive configuration.
- **Extend everything** from model handlers to data processors by updating schemas and wiring your own implementations.
- **Give models superpowers** with MCP (Model Context Protocol) – Connect your AI to local tools, APIs, and databases through a standardized protocol.

## The Power of MCP (Model Context Protocol)

LlamaFarm supports **MCP servers** – a standardized way to give AI models access to external tools and capabilities. Instead of limiting your AI to text generation, you can connect it to filesystems, databases, APIs, calendars, and custom business logic.

### What Makes MCP Powerful

**Per-Model Tool Access Control**: Different models can have different capabilities. Give your production model access to read-only tools while your development model can modify data.

**Multiple Transport Types**: Connect to tools running as:
- **Local processes** (STDIO) – Python scripts, Node.js servers
- **HTTP APIs** – Remote services with standard REST endpoints
- **SSE streams** – Server-Sent Events for real-time data

**Persistent Sessions**: LlamaFarm maintains long-lived connections to MCP servers, avoiding reconnection overhead and improving performance.

### Configuration Example

```yaml
# Define available MCP servers
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "/Users/myuser/documents"

    - name: database
      transport: http
      base_url: http://localhost:8080/mcp
      headers:
        Authorization: Bearer ${env:DB_TOKEN}

# Configure which models can use which servers
runtime:
  models:
    - name: research-assistant
      provider: openai
      model: gpt-4
      # This model can access filesystem only
      mcp_servers:
        - filesystem

    - name: data-analyst
      provider: openai
      model: gpt-4
      # This model can access database only
      mcp_servers:
        - database

    - name: general-chat
      provider: ollama
      model: llama3.1:8b
      # Empty list = no MCP access (safer)
      mcp_servers: []
```

### Real-World Use Cases

- **Code Analysis**: Give models access to your local file system to read and analyze code
- **Data Queries**: Connect to databases and let AI write and execute SQL queries
- **Calendar Management**: Integrate with calendar APIs for scheduling and meeting coordination
- **API Integration**: Connect to external services (weather, CRM, ticketing systems)
- **Custom Business Logic**: Expose internal tools through your own MCP servers

## Choose Your Own Adventure

| Get Started | Go Deeper | Build Your Own |
| ----------- | --------- | -------------- |
| [Quickstart](./quickstart/index.md) – install, init, chat, ingest your first dataset. | [Core Concepts](./concepts/index.md) – architecture, sessions, and components. | [Extending LlamaFarm](./extending/index.md) – add runtimes, stores, parsers, and CLI commands. |
| [Designer Web UI](./designer/index.md) – visual interface for project management. | [Configuration Guide](./configuration/index.md) – schema-driven project settings. | [RAG Guide](./rag/index.md) – strategies, processing pipelines, and monitoring. |
| [CLI Reference](./cli/index.md) – command matrix and examples. | [Models & Runtime](./models/index.md) – configure AI models and providers. | [Prompts](./prompts/index.md) – prompt engineering and management. |

## Philosophy

- **Local-first, cloud-aware** – everything works offline, yet you can point at remote runtimes when needed.
- **Configuration over code** – projects are reproducible because behaviour lives in `llamafarm.yaml`.
- **Composable modules** – RAG, prompts, and runtime selection work independently but integrate cleanly.
- **Flexible interfaces** – Use the CLI for automation, the Designer for visual management, or the REST API for custom integrations.
- **Open for extension** – documentation includes patterns for registering new providers, stores, and utilities.

:::tip Prefer Visual Tools?
The **Designer Web UI** provides a browser-based interface for managing projects, uploading datasets, and testing your AI—all without touching the command line. It's automatically available at `http://localhost:7724` when you run `lf start` (or `http://localhost:3123` if using Docker Compose directly). [Learn more →](./designer/index.md)
:::

## 🎥 In-Depth Tutorial

**Complete Walkthrough (7 minutes):** https://youtu.be/HNnZ4iaOSJ4

Watch a comprehensive demonstration of LlamaFarm's features including project setup, dataset ingestion, RAG queries, and configuration options.

---

Ready to build? Start with the [Quickstart](./quickstart/index.md) and keep the CLI open in another terminal.
