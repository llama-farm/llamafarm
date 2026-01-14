# Subsystem Registry

Defines all subsystems, their tech stacks, and skill mappings for the LlamaFarm monorepo.

---

## Shared Language Skills

| Skill | Directory | Used By | Files |
|-------|-----------|---------|-------|
| Python | `python-skills/` | server, rag, runtime, config, common | SKILL.md, patterns.md, async.md, typing.md, testing.md, error-handling.md, security.md |
| Go | `go-skills/` | cli | SKILL.md, patterns.md, concurrency.md, error-handling.md, testing.md, security.md |
| TypeScript | `typescript-skills/` | designer, electron | SKILL.md, patterns.md, typing.md, testing.md, security.md |
| React | `react-skills/` | designer | SKILL.md, components.md, hooks.md, state.md, performance.md, security.md |

---

## Subsystem Definitions

### CLI

| Property | Value |
|----------|-------|
| **Name** | cli |
| **Path** | `cli/` |
| **Output** | `cli-skills/` |
| **Language** | Go 1.24 |
| **Frameworks** | Cobra, Bubbletea, Lipgloss |
| **Links To** | `go-skills/` |
| **Generates** | SKILL.md, cobra.md, bubbletea.md, performance.md |

### Server

| Property | Value |
|----------|-------|
| **Name** | server |
| **Path** | `server/` |
| **Output** | `server-skills/` |
| **Language** | Python 3.12 |
| **Frameworks** | FastAPI, Celery, Pydantic, Uvicorn |
| **Links To** | `python-skills/` |
| **Generates** | SKILL.md, fastapi.md, celery.md, pydantic.md, performance.md |

### RAG

| Property | Value |
|----------|-------|
| **Name** | rag |
| **Path** | `rag/` |
| **Output** | `rag-skills/` |
| **Language** | Python 3.11 |
| **Frameworks** | LlamaIndex, ChromaDB, Celery |
| **Links To** | `python-skills/` |
| **Generates** | SKILL.md, llamaindex.md, chromadb.md, celery.md, performance.md |

### Universal Runtime

| Property | Value |
|----------|-------|
| **Name** | runtime |
| **Path** | `runtimes/universal/` |
| **Output** | `runtime-skills/` |
| **Language** | Python 3.11 |
| **Frameworks** | PyTorch, Transformers, FastAPI |
| **Links To** | `python-skills/` |
| **Generates** | SKILL.md, pytorch.md, transformers.md, fastapi.md, performance.md |

### Designer

| Property | Value |
|----------|-------|
| **Name** | designer |
| **Path** | `designer/` |
| **Output** | `designer-skills/` |
| **Language** | TypeScript |
| **Frameworks** | React 18, TanStack Query, TailwindCSS, Radix UI, Vite |
| **Links To** | `typescript-skills/`, `react-skills/` |
| **Generates** | SKILL.md, tanstack-query.md, tailwind.md, radix.md, performance.md |

### Electron App

| Property | Value |
|----------|-------|
| **Name** | electron |
| **Path** | `electron-app/` |
| **Output** | `electron-skills/` |
| **Language** | TypeScript |
| **Frameworks** | Electron 28, Electron Vite |
| **Links To** | `typescript-skills/` |
| **Generates** | SKILL.md, electron.md, security.md, performance.md |

### Config

| Property | Value |
|----------|-------|
| **Name** | config |
| **Path** | `config/` |
| **Output** | `config-skills/` |
| **Language** | Python 3.11 |
| **Frameworks** | Pydantic, JSONSchema |
| **Links To** | `python-skills/` |
| **Generates** | SKILL.md, pydantic.md, jsonschema.md |

### Common

| Property | Value |
|----------|-------|
| **Name** | common |
| **Path** | `common/` |
| **Output** | `common-skills/` |
| **Language** | Python 3.10 |
| **Frameworks** | HuggingFace Hub |
| **Links To** | `python-skills/` |
| **Generates** | SKILL.md, huggingface.md |

---

## File Generation Order

1. **Phase 1: Shared Language Skills** (can run in parallel)
   - python-skills/
   - go-skills/
   - typescript-skills/
   - react-skills/

2. **Phase 2: Subsystem Skills** (can run in parallel, after Phase 1)
   - cli-skills/
   - server-skills/
   - rag-skills/
   - runtime-skills/
   - designer-skills/
   - electron-skills/
   - config-skills/
   - common-skills/

---

## Checklist Item Format

Each checklist file should use this format for items:

```markdown
### {Check Name}

**What to check**: {Description}

**Search pattern**:
```bash
grep -rE "{pattern}" --include="*.{ext}" {path}/
```

**Pass criteria**: {What constitutes passing}

**Fail criteria**: {What constitutes failing}

**Severity**: Critical | High | Medium | Low

**Recommendation**: {How to fix violations}
```

---

## Key Files to Analyze Per Subsystem

| Subsystem | Key Files |
|-----------|-----------|
| CLI | `cli/go.mod`, `cli/cmd/**/*.go`, `cli/internal/**/*.go` |
| Server | `server/pyproject.toml`, `server/src/**/*.py` |
| RAG | `rag/pyproject.toml`, `rag/src/**/*.py` |
| Runtime | `runtimes/universal/pyproject.toml`, `runtimes/universal/src/**/*.py` |
| Designer | `designer/package.json`, `designer/src/**/*.tsx`, `designer/src/**/*.ts` |
| Electron | `electron-app/package.json`, `electron-app/src/**/*.ts` |
| Config | `config/pyproject.toml`, `config/src/**/*.py` |
| Common | `common/pyproject.toml`, `common/src/**/*.py` |
