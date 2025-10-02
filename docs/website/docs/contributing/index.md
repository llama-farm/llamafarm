---
title: Contributing
sidebar_position: 12
---

# Contributing

We love contributions—from bug fixes and docs edits to new providers and RAG components. This page summarizes expectations and points you to detailed guides.

## Development Basics

- **Python**: format with `uv run ruff check --fix .`; run tests via `uv run --group test python -m pytest` in relevant packages.
- **Go (CLI)**: run `go fmt ./...`, `go vet ./...`, and `go test ./...` inside `cli/`.
- **Docs**: run `nx build docs` to ensure the site compiles without broken links.

## Workflow

1. Fork or create a branch (e.g., `feat/new-provider`).
2. Make changes—code, schema, docs.
3. Run tests/lint commands; capture results for the PR description.
4. Open a draft PR with a summary, rationale, and testing notes.
5. Iterate with reviewers; keep commits tidy (Conventional Commit style is preferred).

## Updating Schemas

If you modify `config/schema.yaml` or `rag/schema.yaml`:

```bash
cd config
./generate-types.sh
cd ..
```

Re-run relevant tests to ensure generated code compiles and integrations still work.

## Adding Documentation

- Place new pages under the most relevant section (`quickstart`, `cli`, `configuration`, `examples`, etc.).
- Use `index.md` inside directories for Docusaurus categories.
- Keep tone concise and practical; update cross-links as needed.

## Issue Labels

- `good first issue` – scoped tasks ideal for newcomers.
- `help wanted` – contributions welcome; comment if you intend to pick one up.
- `breaking change` – coordinate with maintainers before merging.

## Community

- [Discord](https://discord.gg/RrAUXTCVNF) for real-time chat.
- [Discussions](https://github.com/llama-farm/llamafarm/discussions) for RFCs or bigger feature proposals.
- Weekly sync details are pinned in Discord.

Thanks for helping grow LlamaFarm!
