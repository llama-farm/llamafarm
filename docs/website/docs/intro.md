---
sidebar_position: 1
sidebar_label: Start Here
---

# Welcome to LlamaFarm

LlamaFarm helps you ship retrieval-augmented and agentic AI apps from your laptop to production. It is fully open-source and intentionally extendable—swap model providers, vector stores, parsers, and CLI workflows without rewriting your project.

## What You Can Do Today

- **Prototype locally** with Ollama or any OpenAI-compatible runtime (vLLM, Together, custom gateways).
- **Ingest and query documents** using configurable RAG pipelines defined entirely in YAML.
- **Automate workflows** with a single CLI (`lf`) that manages projects, datasets, and chat interactions.
- **Extend everything** from model handlers to data processors by updating schemas and wiring your own implementations.

## Choose Your Own Adventure

| Get Started | Go Deeper | Build Your Own |
| ----------- | --------- | -------------- |
| [Quickstart](./quickstart/index.md) – install, init, chat, ingest your first dataset. | [Core Concepts](./concepts/index.md) – architecture, sessions, and components. | [Extending LlamaFarm](./extending/index.md) – add runtimes, stores, parsers, and CLI commands. |
| [CLI Reference](./cli/index.md) – command matrix and examples. | [Configuration Guide](./configuration/index.md) – schema-driven project settings. | [RAG Guide](./rag/index.md) – strategies, processing pipelines, and monitoring. |

## Philosophy

- **Local-first, cloud-aware** – everything works offline, yet you can point at remote runtimes when needed.
- **Configuration over code** – projects are reproducible because behaviour lives in `llamafarm.yaml`.
- **Composable modules** – RAG, prompts, and runtime selection work independently but integrate cleanly.
- **Open for extension** – documentation includes patterns for registering new providers, stores, and utilities.

Ready to build? Start with the [Quickstart](./quickstart/index.md) and keep the CLI open in another terminal.
