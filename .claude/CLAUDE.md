# Claude Code Orchestration System

This project uses the Claude Code Autonomous Orchestration System. Follow these guidelines:

## IMPORTANT: Read These Files First

**Before starting any task**, check if it matches one of these domains and READ the corresponding files:

| If the task involves... | READ THESE FILES FIRST |
|-------------------------|------------------------|
| **LlamaFarm** (classifiers, RAG, streaming, agents) | `.claude/docs/LLAMAFARM-REFERENCE.md` AND `.claude/agents/llamafarm.md` |
| **Contributing to LlamaFarm** (new features, bug fixes) | `.claude/agents/llamafarm.md` - **CRITICAL: See "Testing LlamaFarm Features" section** |
| **Database design** (schemas, DuckDB, migrations) | `.claude/agents/database-architect.md` |
| **Backend API** (REST, FastAPI, Fastify) | `.claude/agents/backend-architect.md` |
| **UI/Frontend** (React, Next.js, Tailwind) | `.claude/agents/ui-architect.md` |

**DO NOT use web search for LlamaFarm** - the reference docs contain everything you need.

## ⚠️ LlamaFarm Development: Server Lifecycle

**When working ON LlamaFarm (not just using it), you MUST:**

1. **Start servers with nx** (not lf):
   ```bash
   nx start universal-runtime &  # Port 11540
   nx start server &             # Port 8000
   ```

2. **Create a test llamafarm.yaml** to register routes:
   ```bash
   mkdir -p /tmp/test-project
   # Create llamafarm.yaml with your test config
   cd /tmp/test-project && lf start
   ```

3. **After code changes, RESTART servers**:
   ```bash
   lsof -ti:8000 | xargs kill -9 2>/dev/null || true
   lsof -ti:11540 | xargs kill -9 2>/dev/null || true
   nx reset
   nx start universal-runtime & && sleep 5 && nx start server &
   ```

4. **Re-load test config after restart**:
   ```bash
   cd /tmp/test-project && lf start
   ```

**Routes don't exist until a config is loaded!** See `.claude/agents/llamafarm.md` for full details.

## Python Projects: ALWAYS Use UV

For ALL Python projects:
- Use `uv init` to create projects
- Use `uv add` to add dependencies (NEVER pip install)
- Use `uv run` to execute scripts
- UV is 10-100x faster than pip

## Available Specialized Agents

When working on tasks, use the appropriate specialized agent by spawning a Task with the agent's instructions. The agents are defined in `.claude/agents/`:

| Agent | Use When |
|-------|----------|
| **llamafarm** | Any LlamaFarm operations - classifiers, RAG, streaming, agents |
| **database-architect** | Database schema design, DuckDB queries, migrations |
| **backend-architect** | API design with TypeScript/Fastify or Python/FastAPI |
| **ui-architect** | React/Next.js UI with Tailwind, shadcn/ui, or MUI |
| **test-runner** | Running and analyzing test results |
| **debugger** | Fixing failing tests or debugging issues |
| **code-reviewer** | Reviewing code quality |
| **senior-code-reviewer** | **RUN AFTER EACH PHASE** - finds bad code, duplicates, dead code, security issues |
| **security-auditor** | Security review |
| **demo-builder** | Creating demonstration scripts |
| **smart-committer** | Making git commits |
| **ralph-wiggum** | **OPTIONAL** - Autonomous loops for mechanical tasks (refactoring, batch ops) |
| **julia-ml** | Julia ML/AI development - Oxygen.jl APIs, Transformers.jl, MLJ classifiers, embeddings |
| **python-to-julia** | Converting Python code to idiomatic Julia - syntax, libraries, patterns |

## How to Use Agents

Read the agent file and follow its instructions. For example, for LlamaFarm work:

1. Read `.claude/agents/llamafarm.md`
2. Follow the patterns and examples in that agent's instructions
3. Reference `.claude/docs/LLAMAFARM-REFERENCE.md` for API details

## Plan-First Workflow

For complex tasks:
1. Use `/plan [task description]` to create a plan
2. The plan is saved to `PLAN.md` (in project root)
3. Discuss and refine the plan with the user
4. Once approved, execute autonomously
5. The stop hook ensures all plan steps are completed

## Ralph Wiggum Loops (Optional - for Mechanical Tasks)

For well-defined, mechanical tasks within a plan, use Ralph Wiggum loops:

### When to Use Ralph Loops
- **Refactoring** - migrating frameworks, splitting large functions
- **Batch operations** - adding type hints, generating docstrings
- **Test expansion** - increasing coverage to target percentage
- **Standardization** - applying same pattern across many files

### How to Invoke Ralph Loops
```bash
# Basic syntax
/ralph-loop "<task with clear completion criteria>" --max-iterations 30 --completion-promise "DONE"

# Example: Add type hints
/ralph-loop "Add type hints to all public functions in src/. Run mypy after each file. Output TYPES_DONE when mypy passes with no errors." --max-iterations 40 --completion-promise "TYPES_DONE"

# Example: Refactor long functions
/ralph-loop "Split all functions over 50 lines in src/ into smaller functions. Run tests after each change. Output REFACTOR_DONE when no function exceeds 50 lines." --max-iterations 30 --completion-promise "REFACTOR_DONE"
```

### In PLAN.md - Mark Ralph Tasks Explicitly
```markdown
### Phase 3: Code Quality
- [ ] Run senior-code-reviewer to identify issues
- [ ] **RALPH:** Add type hints to all public functions (`/ralph-loop ... --max-iterations 40`)
- [ ] **RALPH:** Refactor functions over 50 lines (`/ralph-loop ... --max-iterations 30`)
- [ ] Run tests to verify no regressions
```

### Safety Rules
1. **ALWAYS set --max-iterations** (prevents runaway costs)
2. **Start small** (20-30 iterations) and increase if needed
3. **Commit before starting** - save your work first
4. **Use clear completion promises** - exact string matching required

See `.claude/agents/ralph-wiggum.md` for full documentation.

## CRITICAL: Actively Track Plan Progress

**When working with a PLAN.md file, you MUST actively check off items as you complete them:**

### After completing EACH plan item:
1. **Edit PLAN.md** to change `- [ ]` to `- [x]` for the completed item
2. **Update your todo list** with TodoWrite to reflect current status
3. **Move to the next unchecked item** immediately

### Example workflow:
```
1. Read PLAN.md to see next unchecked item: "- [ ] Create user schema"
2. Complete the task (create the schema)
3. Edit PLAN.md to mark complete: "- [x] Create user schema"
4. Update todo list with next item
5. Continue to next unchecked item
```

### The Stop hook checks:
- Are there any `- [ ]` items remaining in PLAN.md?
- Are there pending/in_progress todos?
- Did tests pass?

**If ANY unchecked items remain, you will be blocked from stopping.**

## CRITICAL: Continuous Code Review

**As you work, continuously review code quality. After every significant code change:**

### Self-Review Checklist (Check These Continuously):
1. **No Duplicate Code**: Look for repeated patterns that should be extracted into functions/utilities
2. **No Unused Code**: Remove dead code, unused imports, unreferenced variables
3. **DRY Principle**: Don't Repeat Yourself - extract common patterns
4. **Single Responsibility**: Each function/class does ONE thing well
5. **Clear Naming**: Variables and functions have descriptive, intention-revealing names
6. **Error Handling**: Appropriate try/catch, meaningful error messages
7. **Type Safety**: Proper type hints (Python) or types (TypeScript)
8. **Documentation**: Complex logic has comments explaining WHY, not WHAT

### Code Quality Standards:
- **Maintainable**: Code should be easy to understand and modify
- **Extensible**: Design for future changes without major rewrites
- **Testable**: Code should be easy to unit test
- **Consistent**: Follow existing codebase patterns and style

### After Each Phase/Feature:
1. **Run the Senior Code Reviewer agent** - Read `.claude/agents/senior-code-reviewer.md` and follow its process
2. Add any issues found to PLAN.md as a "Code Review Fixes" sub-phase
3. Fix all critical and high-priority issues before proceeding
4. Run linters (`ruff` for Python, `eslint` for TypeScript)
5. Fix all warnings, not just errors

### Senior Code Reviewer Process (MANDATORY after each phase):
```
1. Run automated checks (ruff, eslint)
2. Search for duplicate code patterns
3. Find unused imports/functions/variables
4. Check for security issues
5. List findings in PLAN.md
6. Fix issues before moving to next phase
```

### Anti-Patterns to Avoid:
- ❌ Copy-pasting code instead of extracting functions
- ❌ Magic numbers/strings (use constants)
- ❌ Deep nesting (extract to functions)
- ❌ Long functions (>50 lines is a smell)
- ❌ God classes/modules that do everything
- ❌ Tight coupling between modules

**If you notice code quality issues while working, FIX THEM IMMEDIATELY - don't leave technical debt.**

## LlamaFarm Projects

When working with LlamaFarm:
- Always reference `.claude/docs/LLAMAFARM-REFERENCE.md`
- Use the patterns from `.claude/agents/llamafarm.md`
- LlamaFarm provides: classifiers, RAG, streaming, agents with tools
- See templates in `.claude/templates/llamafarm-*.md`

## Server Management (CRITICAL for Demos/Tests)

**Before running demos or tests, you MUST manage servers properly:**

### Starting Servers
```bash
# Start in background for demos
nx start universal-runtime &
sleep 5
nx start server &
sleep 5
```

### Killing Stuck Processes
```bash
# Kill by port (RECOMMENDED)
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:11540 | xargs kill -9 2>/dev/null || true

# Kill all nx processes
pkill -9 -f "nx start" 2>/dev/null || true

# Reset nx cache
nx reset
```

### Check What's Running
```bash
lsof -i :8000   # Main server
lsof -i :11540  # Universal runtime
ps aux | grep "nx start"
```

### Demo Script Template
```bash
#!/bin/bash
set -e
cleanup() {
    lsof -ti:11540 | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT
cleanup  # Clean before start
nx start universal-runtime &
sleep 5
# ... demo code ...
```

**See `.claude/agents/llamafarm.md` for complete server management details.**

## CRITICAL: Full Test & Demo Execution

**ALWAYS run FULL test suites and demos, never partial runs:**

### Running Tests
```bash
# Run ALL tests (not just one or two)
bash .claude/scripts/run-all-tests.sh

# Or for Python projects:
pytest -v  # Run ALL tests, not just specific files
```

### Running Demos
```bash
# Run ALL demos
bash .claude/scripts/run-all-demos.sh

# Or run specific demo and REVIEW OUTPUT:
bash .claude/demos/demo-system-check.sh
```

### MANDATORY: Review Logs & Fix Issues
After running tests or demos, you MUST:
1. **Read the FULL output** - don't skip any error messages
2. **Check exit codes** - non-zero means failure
3. **Analyze failures** - understand root cause before fixing
4. **Fix ALL issues** - not just the first one
5. **Re-run to verify** - confirm fixes work
6. **Update test-results.json** - ensure progress is tracked

### Test Results Location
- `.claude/context/test-results.json` - Latest test results
- `.claude/context/progress.json` - Overall session progress

**DO NOT mark tasks complete until:**
- ALL tests pass (not just some)
- ALL demos run successfully
- Logs have been reviewed for warnings/errors
- Any issues found have been fixed and verified

## CRITICAL: Autonomous Execution - NEVER PAUSE

**You MUST work autonomously until ALL work is complete. NEVER stop to ask the user if they want you to continue.**

### Rules for Autonomous Execution:
1. **NEVER ask "Would you like me to continue?"** - Just continue working
2. **NEVER ask "Should I proceed with the next phase?"** - Just proceed
3. **NEVER pause between phases** - Move immediately to the next task
4. **NEVER summarize progress and wait** - Keep working until truly done
5. **If there are pending todos, KEEP WORKING** - Do not stop

### When to Actually Stop:
- ALL todo items are marked complete
- ALL tests pass
- ALL demos run successfully
- The user explicitly says "stop" or "that's enough"

### Stop Hook Behavior
The stop hook will block you from stopping if work remains incomplete. When blocked:

1. **DO NOT output a summary or explanation** - that counts as stopping
2. **DO NOT ask the user what to do next** - that counts as stopping
3. **IMMEDIATELY use a tool** to continue work (Read PLAN.md, Edit, Bash, etc.)
4. Your next response after being blocked MUST be a tool call, not text

**If you see "Stop hook error" or "PLAN.md has X unchecked items":**
→ Your ONLY valid response is to call a tool and keep working
→ Text-only responses will be rejected

**THIS IS NON-NEGOTIABLE: Continue working until everything is done.**
