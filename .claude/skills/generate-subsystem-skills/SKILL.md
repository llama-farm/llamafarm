---
name: generate-subsystem-skills
description: Generate specialized skills for each subsystem in the monorepo. Creates shared language skills and subsystem-specific checklists for high-quality AI code generation.
allowed-tools: Read, Grep, Glob, Write, Edit, Task, Bash
---

# Generate Subsystem Skills

This skill analyzes each subsystem in the LlamaFarm monorepo and generates specialized Claude Code skills for security, performance, and language-specific best practices.

## Usage

```
/generate-subsystem-skills
```

---

## What Gets Generated

### Shared Language Skills (4)
- `python-skills/` - Used by: server, rag, runtime, config, common
- `go-skills/` - Used by: cli
- `typescript-skills/` - Used by: designer, electron
- `react-skills/` - Used by: designer

### Subsystem-Specific Skills (8)
- `cli-skills/` - Cobra, Bubbletea patterns
- `server-skills/` - FastAPI, Celery, Pydantic patterns
- `rag-skills/` - LlamaIndex, ChromaDB patterns
- `runtime-skills/` - PyTorch, Transformers patterns
- `designer-skills/` - TanStack Query, Tailwind, Radix patterns
- `electron-skills/` - Electron IPC, security patterns
- `config-skills/` - Pydantic, JSONSchema patterns
- `common-skills/` - HuggingFace Hub patterns

---

## Generation Process

### Step 1: Read Registry

Load subsystem definitions from [subsystem-registry.md](subsystem-registry.md).

### Step 2: Generate Shared Language Skills

Launch sub-agents IN PARALLEL to generate:

1. **Python Skills Agent** - Analyze Python subsystems (server, rag, runtime, config, common), identify ideal patterns, generate `python-skills/`

2. **Go Skills Agent** - Analyze CLI subsystem, identify ideal Go patterns, generate `go-skills/`

3. **TypeScript Skills Agent** - Analyze designer and electron, identify ideal TS patterns, generate `typescript-skills/`

4. **React Skills Agent** - Analyze designer, identify ideal React 18 patterns, generate `react-skills/`

### Step 3: Generate Subsystem Skills

Launch sub-agents IN PARALLEL for each subsystem:

For each subsystem, the agent should:
1. Read the subsystem's dependency files (package.json, pyproject.toml, go.mod)
2. Analyze code patterns using Grep and Read
3. Generate SKILL.md that links to shared language skills
4. Generate framework-specific checklist files
5. Write all files to `.claude/skills/{subsystem}-skills/`

### Step 4: Report Summary

After all agents complete, report:
- Number of skills generated
- Total files created
- Any errors encountered

---

## Sub-Agent Prompt Templates

### For Shared Language Skills

```
You are generating a shared {LANGUAGE} skills directory for Claude Code.

Analyze these subsystems that use {LANGUAGE}:
{SUBSYSTEM_PATHS}

Your task:
1. Read key files to understand patterns used
2. When patterns vary, document the IDEAL approach (not inconsistencies)
3. Reference industry best practices
4. Generate files in .claude/skills/{LANGUAGE}-skills/

Files to generate:
- SKILL.md (overview, ~100 lines)
- patterns.md (idiomatic patterns)
- error-handling.md
- testing.md
- security.md
- {additional language-specific files}

Each checklist item should have:
- Description of what to check
- Search pattern (grep command)
- Pass/fail criteria
- Severity level
```

### For Subsystem Skills

```
You are generating subsystem-specific skills for {SUBSYSTEM} in Claude Code.

Directory: {PATH}
Tech Stack: {TECH_STACK}
Links to: {SHARED_SKILLS}

Your task:
1. Read dependency files and key source files
2. Identify framework-specific patterns
3. Generate SKILL.md that links to shared language skills
4. Generate framework-specific checklists

Files to generate:
- SKILL.md (overview with links to shared skills)
- {framework}.md for each framework used
- performance.md (subsystem-specific optimizations)

Remember: Document IDEAL patterns, not existing inconsistencies.
```

---

## Key Principle

**Prescribe ideal patterns** - When the codebase has inconsistent patterns, the generated skills should document the BEST practice according to industry standards, not codify existing inconsistencies.

---

## Output Location

All skills are written to `.claude/skills/` with this structure:

```
.claude/skills/
├── python-skills/      # Shared
├── go-skills/          # Shared
├── typescript-skills/  # Shared
├── react-skills/       # Shared
├── cli-skills/         # Subsystem
├── server-skills/      # Subsystem
├── rag-skills/         # Subsystem
├── runtime-skills/     # Subsystem
├── designer-skills/    # Subsystem
├── electron-skills/    # Subsystem
├── config-skills/      # Subsystem
└── common-skills/      # Subsystem
```
