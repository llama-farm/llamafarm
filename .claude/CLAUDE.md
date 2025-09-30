# LlamaFarm – Comprehensive Guide for LLM Assistants

This document summarizes everything an automated collaborator needs to get LlamaFarm running, extend it, and keep configuration accurate.

## 1. Prerequisites & Fast Start
1. **Install Docker** – required for auto-starting the API + RAG worker.
2. **Install Ollama** – current default runtime; download from https://ollama.com/download.
3. **Install the CLI (lf)**
   ```bash
   # macOS / Linux
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```
   - Windows: download `lf.exe` from the latest release and add it to PATH.
4. **Adjust Ollama context window** – open Ollama → Settings → Advanced → set context window size (e.g., 100k tokens) to match production expectations.

### First Run
```bash
lf init my-project        # generates llamafarm.yaml via server template
lf start                  # starts FastAPI + Celery workers in Docker, opens chat TUI
```
- Use `Ctrl+C` to exit the chat UI.
- Once running, the CLI provides:
  - Dataset lifecycle: `lf datasets create|upload|process`
  - Retrieval: `lf rag query --database ...`
  - Health checks: `lf rag health`, `lf rag stats`
  - One-off chat: `lf chat [...flags]`

## 2. Repository Layout (top-level):
- `README.md` – CLI-first quickstart, extensibility highlights, testing commands.
- `config/` – schema + generated types for `llamafarm.yaml`.
- `server/` – FastAPI app, Celery task definitions.
- `rag/` – ingestion/parsing utilities, Celery worker entry.
- `cli/` – Go-based CLI (`lf`) built with Cobra.
- `docs/website/` – Docusaurus doc site (`nx build docs`).
- `.claude/` – this helper file and any future LLM-specific directions.

## 3. Key Documentation Pages (relative paths):
- Quickstart onboarding: `docs/website/docs/quickstart/index.md`
- CLI reference: `docs/website/docs/cli/index.md`
- Configuration guide: `docs/website/docs/configuration/index.md`
- RAG guide: `docs/website/docs/rag/index.md`
- Models/runtime: `docs/website/docs/models/index.md`
- Extensibility overview: `docs/website/docs/extending/index.md`
- Examples: `docs/website/docs/examples/index.md`
- Troubleshooting: `docs/website/docs/troubleshooting/index.md`

## 4. Configuration (`llamafarm.yaml`)
Schema files:
- `config/schema.yaml` – top-level project schema.
- `rag/schema.yaml` – inlined for RAG-specific fields (databases, strategies, parsers, extractors).

Required sections:
- `version` – currently `v1`.
- `name`, `namespace` – identify the project/tenant.
- `runtime` – must include `provider`, `model`, and for non-default hosts, `base_url`/`api_key`.
- `prompts` – list of `{role, content}` messages.
- `rag` – `databases` + `data_processing_strategies` definitions.
- `datasets` (optional) – keep dataset metadata in sync.

Reference docs: `docs/website/docs/configuration/index.md` and `docs/website/docs/configuration/example-configs.md`.

## 5. Editing Schemas & Types
When you change `config/schema.yaml` or `rag/schema.yaml`:
```bash
cd config
./generate-types.sh
```
Outputs:
- `config/datamodel.py` (Pydantic models)
- `config/config_types.go` (Go structs for CLI)

Update the CLI/server code to handle new fields, then adjust docs accordingly.

## 6. Extending Components
### Runtime Providers
1. Add provider enum value in `config/schema.yaml` (`runtime.provider`).
2. Regenerate types (`config/generate-types.sh`).
3. Update runtime selection logic:
   - Server: `server/services/runtime_service.py` (or relevant module).
   - CLI: wherever runtime resolution occurs (e.g., `cli/cmd/config/types.go` consumers).
4. Document usage in `docs/website/docs/models/index.md`.

### RAG Stores / Parsers / Extractors
1. Implement the component inside `rag/` (e.g., new store class, parser implementation).
2. Register it with the ingestion pipeline.
3. Update `rag/schema.yaml` definitions:
   - `databaseDefinition.type`
   - Parser/extractor enums & config schemas.
4. Regenerate types and update docs (`docs/website/docs/rag/index.md`).

### CLI Commands
- Add Cobra command under `cli/cmd/`, hook it via `rootCmd.AddCommand(...)`.
- Follow patterns for config resolution (`config.GetServerConfig`) and server auto-start (`ensureServerAvailable`).
- Write tests (`*_test.go`) and update the CLI reference documentation.

## 7. Common CLI Commands
```bash
# Dataset lifecycle
lf datasets create -s pdf_ingest -b main_db research
lf datasets upload research ./examples/fda_rag/files/*.pdf
lf datasets process research

# Retrieval & health
lf rag query --database main_db "Which letters mention clinical trials?"
lf rag health
lf rag stats

# Chat
lf chat "Summarize neural scaling laws"
lf chat --no-rag "Explain attention mechanisms"
lf chat --curl "What models are configured?"
```

## 8. Testing & Validation
- Server & RAG Python tests: `uv run --group test python -m pytest`
- RAG-specific tests: `uv run pytest tests/`
- CLI: `go test ./...`
- Docs build: `nx build docs` *(clear `.nx` cache if you hit sqlite disk I/O errors)*

## 9. Upgrade / Development Notes
- Always run research/plan steps from `.agents/commands/` before making changes.
- Keep docs in sync with behaviour—update Docusaurus pages when workflows/schema change.
- Never commit secrets; use local `.env` and update `.env.example` for new variables.
- When adding user-facing features, ensure README + docs + sample configs all reflect the changes.

## 10. Additional Resources
- Project structure overview: `AGENTS.md`
- Contribution process: `CONTRIBUTING.md`
- Credits: `docs/CREDITS.md`
- Examples: `examples/` directory with ready-made configs (`fda_rag`, `gov_rag`).

Use this guide as the master reference when assisting with LlamaFarm tasks.
