---
title: Examples
sidebar_position: 10
---

# Example Workflows

## 🌟 Featured: Personal Medical Assistant

**A privacy-first, 100% local medical records helper** that lets you understand your health data using AI and evidence-based medical knowledge.

<div style={{position: 'relative', paddingBottom: '56.25%', height: 0, overflow: 'hidden', maxWidth: '100%', marginBottom: '2rem'}}>
  <iframe
    style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}}
    src="https://www.youtube.com/embed/H6WKKzYPLlQ"
    title="Personal Medical Assistant Demo"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowFullScreen>
  </iframe>
</div>

### ✨ Highlights

- 🔒 **Complete Privacy** – All PDF processing in browser, no uploads, PHI stays on device
- 🤖 **Multi-Hop Agentic RAG** – AI orchestrates query generation and knowledge synthesis
- 📚 **125,830 Medical Knowledge Chunks** – From 18 authoritative textbooks (MedRAG dataset)
- ⚡ **Two-Tier AI** – Fast model for queries, capable model for responses
- 💬 **Streaming Chat** – Real-time responses with transparent reasoning

Built with Next.js and LlamaFarm, this example demonstrates how to build privacy-first healthcare applications that keep sensitive data completely local while delivering intelligent, evidence-based insights.

**[📖 Read the Full Guide →](./medical-records-helper.md)**

---

## CLI-Based RAG Examples

The repository ships with interactive demos that highlight different retrieval scenarios. Each example lives under `examples/<folder>` and provides a configuration, sample data, and a script that uses the newest CLI commands (e.g., `lf datasets create`, `lf chat`).

| Folder | Use Case | Highlights |
|--------|----------|------------|
| `large_complex_rag/` | Multi-megabyte Raleigh UDO ordinance PDF | Long-running ingestion, zoning-focused prompts, unique DB/dataset per run. |
| `many_small_file_rag/` | FDA correspondence packet | Several shorter PDFs, quick iteration, letter-specific queries. |
| `mixed_format_rag/` | Blend of PDF/Markdown/HTML/text/code | Hybrid retrieval, multiple parsers/extractors in one pipeline. |
| `quick_rag/` | Two short engineering notes | Rapid smoke test for the environment and CLI. |

## How to Run an Example
```bash
# Build or install the CLI if needed
go build -o lf ./cli

# Run the interactive workflow (press Enter between steps).
# The script automatically scopes the CLI with `lf --cwd examples/<folder>`.
./examples/<folder>/run_example.sh

# Optional: point the script at a different directory that contains the lf binary
./examples/<folder>/run_example.sh /path/to/your/project

# Skip prompts if desired
NO_PAUSE=1 ./examples/<folder>/run_example.sh
```

Each script clones the relevant database entry, creates a unique dataset/database pair, uploads the sample documents, processes them, prints the CLI output verbatim, runs meaningful `lf rag query` and `lf chat` commands, and finishes with a baseline `--no-rag` comparison. Clean-up instructions are printed at the end of each script.

## Manual Command Reference
Use these commands if you prefer to run the workflows yourself (replace `<folder>` with the example you want to explore):
```bash
lf --cwd examples/<folder> datasets create -s <strategy> -b <database> <dataset>
lf --cwd examples/<folder> datasets upload <dataset> examples/<folder>/files/*
lf --cwd examples/<folder> datasets process <dataset>
lf --cwd examples/<folder> rag query --database <database> --top-k 3 --include-metadata --include-score "Your question"
lf --cwd examples/<folder> chat --database <database> "Prompt needing citations"
lf --cwd examples/<folder> chat --no-rag "Same prompt without RAG"
lf --cwd examples/<folder> datasets delete <dataset>
rm -rf examples/<folder>/data/<database>
```

Refer to each example folder’s README for scenario-specific prompts, cleanup suggestions, and contextual background (e.g., why those documents were chosen and what use cases they simulate).
