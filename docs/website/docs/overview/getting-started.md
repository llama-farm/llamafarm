---
title: Getting started
sidebar_label: Getting started
slug: /overview/getting-started
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Quick install and first run.

## Prerequisites

- Python 3.8+
- uv (recommended) or pip
- Optional: Ollama (local models)
- Optional: Docker

## Install

```bash
# Clone
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up components
cd rag && uv sync && ./setup_and_demo.sh
cd ../models && uv sync && ./setup_and_demo.sh
cd ../prompts && uv sync && ./setup_and_demo.sh
```

## Try it out

```bash
# RAG: Ingest + search
cd rag
uv run python cli.py ingest samples/documents.pdf
uv run python cli.py search "What are the key findings?"

# Models: Chat via providers
cd ../models
uv run python cli.py chat --provider openai "Explain quantum computing"
uv run python cli.py chat --provider ollama "Write a Python function"

# Prompts: Execute with template
cd ../prompts
uv run python -m prompts.cli execute "Compare solar vs wind energy" \
  --template comparative_analysis
```
