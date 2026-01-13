# Ralph Wiggum Agent - Autonomous Loop Mode

Ralph Wiggum is an **optional autonomous loop mode** for well-defined, mechanical tasks. Use it when you need persistent iteration without human intervention.

## Installation

**Option 1: Add Anthropic marketplace first (recommended)**
```bash
/plugin marketplace add anthropics/claude-code
/plugin install ralph-wiggum
```

**Option 2: Direct GitHub install**
```bash
/plugin install anthropics/claude-code/plugins/ralph-wiggum
```

**Option 3: Manual installation**
```bash
# Clone the repo
git clone https://github.com/anthropics/claude-code.git /tmp/claude-code

# Run claude with the plugin directory
claude --plugin-dir /tmp/claude-code/plugins/ralph-wiggum
```

**Dependency:** Requires `jq` installed on your system:
```bash
# macOS
brew install jq

# Ubuntu/Debian
sudo apt install jq

# Windows (use WSL or install jq manually)
```

## When to Use Ralph Wiggum

### Good For (USE IT):
- **Large-scale refactoring** - migrating frameworks, upgrading dependencies
- **Batch operations** - adding type hints, generating docstrings, standardizing code
- **Test expansion** - increasing coverage to a target percentage
- **Mechanical fixes** - applying same fix pattern across many files
- **Greenfield scaffolding** - generating boilerplate with clear specs

### Not Good For (USE PLAN.md INSTEAD):
- Complex multi-phase projects with dependencies
- Tasks requiring architectural decisions
- Ambiguous requirements without clear "done" criteria
- Security-sensitive code (authentication, payments)
- Exploratory work needing human judgment

## How Ralph Wiggum Works

1. You provide a prompt with clear completion criteria
2. Claude works on the task
3. When Claude tries to stop, the loop re-injects the same prompt
4. Claude sees its previous work in modified files and git history
5. Loop continues until completion promise or max iterations

```
┌─────────────────────────────────────────┐
│  /ralph-loop "task" --max-iterations 50 │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         ┌───────────────┐
         │  Claude works │
         │   on task     │
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │ Tries to stop │
         └───────┬───────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  ┌──────────┐     ┌──────────────┐
  │ Complete │     │ Not Complete │
  │ (exit)   │     │ (re-inject)  │──┐
  └──────────┘     └──────────────┘  │
                          ▲          │
                          └──────────┘
```

## Commands

### Start a Ralph Loop
```bash
/ralph-loop "<prompt>" --max-iterations <n> --completion-promise "<text>"
```

**Parameters:**
- `<prompt>` - The task description with clear completion criteria
- `--max-iterations <n>` - Safety limit (ALWAYS SET THIS, default: 20)
- `--completion-promise "<text>"` - Phrase that signals completion

### Cancel a Ralph Loop
```bash
/cancel-ralph
```

## Prompt Writing for Ralph

### 1. Clear Completion Criteria (CRITICAL)

❌ **Bad:**
```
Refactor the codebase to be cleaner.
```

✅ **Good:**
```
Refactor all Python files in src/ to:
1. Add type hints to all public functions
2. Extract duplicate code into shared utilities
3. Ensure all functions are < 50 lines

Completion criteria:
- ruff check passes with no errors
- No functions exceed 50 lines
- No duplicate code blocks > 5 lines

Output <promise>REFACTOR_COMPLETE</promise> when done.
```

### 2. Incremental Goals

❌ **Bad:**
```
Migrate the entire app from JavaScript to TypeScript.
```

✅ **Good:**
```
Migrate src/utils/ from JavaScript to TypeScript:

Phase 1: Rename .js files to .ts
Phase 2: Add basic types (no 'any' allowed)
Phase 3: Fix all TypeScript errors
Phase 4: Ensure tests pass

Output <promise>MIGRATION_COMPLETE</promise> when all phases done.
```

### 3. Self-Correction Instructions

✅ **Good:**
```
Add comprehensive test coverage to src/api/:

1. Write tests for each endpoint
2. Run pytest after each file
3. If tests fail, fix the test or code
4. Target: 80% coverage minimum

Check coverage with: pytest --cov=src/api --cov-report=term

Output <promise>TESTS_COMPLETE</promise> when coverage >= 80%.
```

### 4. Escape Hatches

Always include fallback instructions:

```
After 15 iterations, if not complete:
- Document what's blocking progress in BLOCKING.md
- List what was attempted
- Suggest alternative approaches
- Output <promise>NEEDS_HELP</promise>
```

## Example Ralph Loops

### Refactoring Example
```bash
/ralph-loop "Refactor all functions in src/ to be under 50 lines. Split large functions into smaller, well-named helper functions. Run ruff check after each change. Output <promise>REFACTOR_DONE</promise> when no function exceeds 50 lines." --max-iterations 30 --completion-promise "REFACTOR_DONE"
```

### Type Hints Example
```bash
/ralph-loop "Add type hints to all public functions in src/. Use specific types, not Any. Run mypy src/ after each file. Fix any type errors. Output <promise>TYPES_DONE</promise> when mypy passes with no errors." --max-iterations 40 --completion-promise "TYPES_DONE"
```

### Test Coverage Example
```bash
/ralph-loop "Increase test coverage in tests/ to 80%. Write tests for uncovered functions. Run pytest --cov after each test file. Output <promise>COVERAGE_DONE</promise> when coverage >= 80%." --max-iterations 50 --completion-promise "COVERAGE_DONE"
```

### Documentation Example
```bash
/ralph-loop "Add docstrings to all public functions and classes in src/. Follow Google docstring format. Include Args, Returns, and Raises sections. Output <promise>DOCS_DONE</promise> when all public APIs are documented." --max-iterations 25 --completion-promise "DOCS_DONE"
```

### Dependency Upgrade Example
```bash
/ralph-loop "Upgrade all dependencies in package.json to latest versions. Run npm test after each upgrade. If tests fail, either fix the code or pin to a working version. Output <promise>UPGRADE_DONE</promise> when all deps are upgraded and tests pass." --max-iterations 40 --completion-promise "UPGRADE_DONE"
```

## Cost Considerations

Ralph loops consume tokens rapidly:
- A 50-iteration loop on a large codebase can cost **$50-100+**
- Always set `--max-iterations` as a safety net
- Start with lower limits (20-30) and increase if needed
- Monitor token usage during loops

## Integration with PLAN.md Workflow

Ralph Wiggum can be used **within** a PLAN.md phase for mechanical sub-tasks:

```markdown
### Phase 3: Code Quality Improvements
- [ ] Run senior-code-reviewer to find issues
- [ ] Fix critical issues manually
- [ ] **USE RALPH:** Refactor long functions (target: <50 lines)
- [ ] **USE RALPH:** Add type hints to all public functions
- [ ] Run tests to verify no regressions
```

When using Ralph within a phase:
1. Complete the Ralph loop
2. Return to PLAN.md
3. Mark the sub-task complete: `- [x] **USE RALPH:** ...`
4. Continue with next item

## Safety Guidelines

1. **Always set --max-iterations** - Never run unlimited loops
2. **Use clear completion promises** - Exact string matching required
3. **Commit before starting** - Save your work first
4. **Monitor progress** - Check in periodically
5. **Test in sandbox first** - For risky refactors, try on a branch

## When to Switch from Ralph to PLAN.md

Switch back to PLAN.md workflow if:
- Task turns out to be more complex than expected
- Architectural decisions are needed
- You hit max iterations without completion
- Human judgment is required for trade-offs
