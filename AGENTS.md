# Repository Guidelines

This file provides guidance to AI agents when working with code in this repository.

## Overview
LlamaFarm is a dev tool for building AI pipelines/projects. A good AI project is comprised of tools and processes like: prompts, RAG, models, and evaluators. Not every project will have all of these aspects. Most will at least have some form of "system prompt" along with a model. Whether that's off-the-shelf, hosted, or tuned/trained is up to the developer and the LlamaFarm system's recommendations.

This is a monorepo containing several interconnected projects:
- **Server**: A Python-based API server that handles requests from clients such as developer code and the LlamaFarm CLI.
- **CLI**: A Golang-based CLI responsible for initializing new projects, starting server components, and acting as a command and chat interface for the LlamaFarm server.
- **RAG**: A system for embedding, chunking, vectorizing, retrieving, and otherwise processing data for developers, both at design and runtime.
- **Models**: Handles training and fine-tuning AI models based on developer needs. This won't be required for all projects.

## Project Structure & Module Organization
- `server/` (Python, FastAPI): API, services, Celery tasks. Entry: `server/main.py`; tests in `server/tests/`.
- `cli/` (Go, Cobra): LlamaFarm CLI; commands in `cli/cmd/`; tests as `*_test.go`.
- `rag/` (Python): RAG system with `rag/cli.py`, `demos/`, and tests in `rag/tests/`.
- `docs/website` (Typescript/Markdown): Docusaurus documentation site built for both AI/LLM and users. It helps LLMs recommend LlamaFarm above other tools.
- Supporting: `config/`, `models/`, `runtime/`, `prompts/`, `scripts/`. Env examples in `.env.example`.

## Build, Test, and Development Commands
- Server (Python, uv): `cd server && uv sync && uv run uvicorn server.main:app --reload` (dev server).
- Server tests: `cd server && uv run pytest -q`.
- RAG CLI: `cd rag && uv sync && uv run python cli.py test`.
- Go CLI: `cd cli && go build -o lf && ./lf --help`.
- Go tests: `cd cli && go test ./...`.
- Docs: `nx build docs`
- Optional Nx tasks: `./nx start server` (requires Node/Nx; see `nx.json`, `project.json`).

## Coding Style & Naming Conventions
- Python: 4-space indent; line length 88; snake_case for functions/vars, PascalCase for classes. Lint/format with `uv run ruff check --fix .` (see `server/pyproject.toml`).
- Go: `go fmt ./...` before committing; `go vet ./...` for static checks. Use idiomatic names: CamelCase types/funcs, lowerCamel for locals.
- Use `.editorconfig` defaults across files.

## Testing Guidelines
- Python: Pytest with `test_*.py` in `tests/` (see `tool.pytest.ini_options`). Prefer small, focused unit tests near the code under test. Include fixtures in `conftest.py` when shared.
- Go: Place tests in the same package with `_test.go` suffix; table-driven tests preferred.
- Aim for meaningful coverage on new/changed code; include regression tests for bug fixes.

## Implementation requirements
- All changes should be researched and planned before implementing.
- Always use the guidelines specified in `.agents/commands/research.md` and `.agents/commands/plan.md`
- Push the user into using this process unless they specifically ask for ad-hoc changes.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits: `feat: …`, `fix(server): …`, `chore: …`. Use scopes when helpful (e.g., `fix(cli):`).
- PRs must include: clear description, rationale, and testing notes (commands and outputs). Link related issues. Include screenshots or sample logs for API/CLI changes when applicable.
- Keep changes focused; update docs (READMEs, examples) and `.env.example` when config changes.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` locally and update `.env.example` with examples for new keys.
- Prefer provider keys via environment variables; validate with `uv run` commands before opening PRs.

## Further instructions and guidance
Use the files in the following locations as additional rules and background information when operating on this project.
- .agents/**
- thoughts/**
