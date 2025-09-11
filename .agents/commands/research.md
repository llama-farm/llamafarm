## Codebase Research Playbook

A fast, reliable way for agents to understand this monorepo before implementing an issue. Follow this checklist to map scope, find the right code, and de‑risk changes.

### Objectives
- **Understand the issue**: clarify scope, inputs/outputs, and success criteria.
- **Locate the code path**: find entry points, flows, configs, and tests.
- **De‑risk**: identify edge cases, cross‑project impacts, and verification steps.

## Quickstart Checklist (15–30 min)
- **Read the issue**: capture user story, acceptance criteria, affected area(s), and constraints.
- **Identify project(s)**: server, cli, rag, models, docs. If unsure, start with Nx graph.
- **Skim module entry points**:
  - `server/main.py` and `server/api/**`
  - `cli/main.go` and `cli/cmd/**`
  - `rag/cli.py` and `rag/**`
  - `docs/website/**`
- **Find existing tests** near the suspected code.
- **Collect commands to run** (build/test/dev) so you can validate later.

## Workspace-first view (Nx preferred)
If Nx tools are available, prefer them to map dependencies and tasks.
- **Graph**: visualize projects and edges.
  - Example: run a project graph or open the visualization tool to see producers/consumers.
- **Show project details**: inspect target configurations (build/test/lint/dev).
- **Docs**: consult relevant Nx docs for generators/tasks if needed.

If Nx tools are not available, manually inspect:
- `nx.json`, `project.json` files at repo and project roots
- `package.json` scripts (if present in subprojects)
- `server/project.json`, `server/pyproject.toml`
- `cli/go.mod`, `cli/project.json`
- `rag/pyproject.toml`, `rag/default_strategies.yaml`

## How to search effectively
- **Start semantic, then go exact**:
  - Use semantic search for “how/where” questions (endpoints, flows, responsibilities).
  - Use exact search (ripgrep/grep) for symbols, function names, routes, or constants.
- **Good semantic queries**:
  - “Where is user authentication handled?”
  - “How are CLI commands registered?”
  - “Where is RAG document ingestion triggered?”
- **Good exact searches (ripgrep examples)**:
```bash
rg --hidden --glob '!**/dist/**' --glob '!**/node_modules/**' "uvicorn|FastAPI" server | cat
rg --hidden "cobra\.Command" cli | cat
rg --hidden "pytest|conftest\.py" server | cat
rg --hidden "project\.json" -n | cat
```
- **Trace the flow**: when you find a hit, read nearby files; follow imports, routes, and callers.
  - Expand outward until you can explain inputs, transformations, side effects, and outputs.

## Project map (what lives where)
- **Server (Python, FastAPI)**
  - Entry: `server/main.py`; APIs in `server/api/**`; services in `server/services/**`.
  - Tests: `server/tests/**`.
  - Dev: `cd server && uv sync && uv run uvicorn server.main:app --reload`.
  - Tests: `cd server && uv run pytest -q`.
- **CLI (Go, Cobra)**
  - Entry: `cli/main.go`; commands in `cli/cmd/**`.
  - Build: `cd cli && go build -o lf && ./lf --help`.
  - Tests: `cd cli && go test ./...`.
- **RAG (Python)**
  - Entry: `rag/cli.py`; components in `rag/components/**`; retrieval/stores under `rag/**`.
  - Dev smoke: `cd rag && uv sync && uv run python cli.py test`.
  - Tests: `cd rag && uv run pytest -q` (if present).
- **Docs**
  - Site under `docs/website/**`.
  - Build: `nx build docs` (if Nx is set up locally).
- **Config & Schemas**
  - Project/config schemas: `config/**`.
  - Prompts/templates: `prompts/**`.
  - Models/training: `models/**`.

## Verify assumptions early
- **Read tests first**: they reveal API shapes, invariants, and edge cases.
- **Check config**: `.env.example`, YAML/TOML under `config/`, and `project.json` targets.
- **Run a thin dev path** (server or CLI) to see runtime behavior before making changes.
- **Ask questions of the user** for any uncertainties, to validate assumptions, or to help make choices.

## Research deliverable (what to write up)
Create a short note in `thoughts/shared/research/<issue_id>_<slug>.md` including:
- **Context**: issue summary and acceptance criteria
- **Affected modules**: files/dirs and why
- **Key flows**: sequence from entry to side effects
- **Risks**: edge cases, cross-project impacts, data migrations
- **Open questions**: unknowns and how to answer them
- **Proposed plan**: minimal, measurable steps to implement and verify

Example template:
```markdown
## Context

## Affected modules

## Key flows

## Risks

## Open questions

## Proposed plan
```

## Citing code and commands
- **When referencing existing code** in your notes or PRs, cite with file path and a tight snippet. If needed, include line numbers from your editor for clarity.
- **When proposing new code**, use fenced code blocks with the language tag.
- **Commands** should use fenced `bash` blocks.

Examples:
```bash
cd server && uv run pytest -q
```

```python
# server/api/example.py
from fastapi import APIRouter
router = APIRouter()
```

## Anti-patterns to avoid
- **Don’t implement before mapping** the code path and tests.
- **Don’t refactor while exploring**; keep research separate from changes.
- **Don’t rely on single search hits**; corroborate via tests and callers.
- **Don’t skip verification**; always plan how you’ll test the change.
- **Don't plan yet**; planning implementation is done via a separate step after research.

## Timebox and check-ins
- **Timebox discovery**: ~5 minutes to produce the research note.
- **Escalate blockers** early with specific questions and file references.

---
If you’re unsure about Nx setup or targets, prefer using Nx workspace tools to inspect projects and graphs before proceeding. When in doubt, document your assumptions in the research note and validate them with a minimal run/test.
