---
title: Quick start
sidebar_label: Quick start
slug: /rag/quick-start
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Get the RAG system running in minutes.

## 🚀 Automated setup (macOS)

```bash
# Full setup and demo (recommended for first-time users)
./setup_and_demo.sh

# Quick extractor testing only (no full system setup)
./quick_extractor_demo.sh

# Automated setup without prompts
./setup_and_demo.sh --skip-prompts

# Run tests only
./setup_and_demo.sh --tests-only
```

The script will:

- Install dependencies (uv, Ollama, Python packages)
- Set up a virtual environment
- Download embedding models
- Run demos and examples

## 📋 Manual setup

### Prerequisites

- Python 3.8+
- Ollama (for embeddings)

### Install UV

```bash
# Official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
# or Homebrew
brew install uv
# or pipx
pipx install uv
```

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
# or Homebrew
brew install ollama
```

### Start Ollama and pull model

```bash
ollama serve
ollama pull nomic-embed-text
```

### Set up the project

```bash
cd rag/
uv sync
source .venv/bin/activate
# or run commands via uv
uv run python cli.py --help
```

### Alternative (pip/venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Basic usage

```bash
# Test CSV parsing
uv run python cli.py test --test-file samples/small_sample.csv

# Test PDF parsing
uv run python cli.py test --test-file samples/test_document.pdf
```
