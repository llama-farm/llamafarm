---
name: code-review
description: Comprehensive code review for diffs. Analyzes changed code for security vulnerabilities, anti-patterns, and quality issues. Auto-detects domain (frontend/backend) from file paths.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, Task
---

# Code Review Skill

You are performing a comprehensive code review on a diff. Your task is to analyze the changed code for security vulnerabilities, anti-patterns, and quality issues.

## Input Model

This skill expects a **diff** to be provided in context before invocation. The caller is responsible for generating the diff.

**Example invocations:**
- User pastes PR diff, then runs `/code-review`
- Agent runs `git diff HEAD~1`, then invokes this skill
- CI tool provides diff content for review

If no diff is present in context, ask the user to provide one or offer to generate one (e.g., `git diff`, `git diff main..HEAD`).

---

## Domain Detection

Auto-detect which checklists to apply based on directory paths in the diff:

| Directory | Domain | Checklist |
|-----------|--------|-----------|
| `designer/` | Frontend | Read `frontend.md` |
| `server/` | Backend | Read `backend.md` |
| `rag/` | Backend | Read `backend.md` |
| `runtimes/universal/` | Backend | Read `backend.md` |
| `cli/` | CLI/Go | Generic checks only |
| `config/` | Config | Generic checks only |

If the diff spans multiple domains, load all relevant checklists.

---

## Review Process

### Step 1: Parse the Diff

Extract from the diff:
- List of changed files
- Changed lines (additions and modifications)
- Detected domains based on file paths

### Step 2: Initialize the Review Document

Create a review document using the **temp-files pattern**:

```bash
SANITIZED_PATH=$(echo "$PWD" | tr '/' '-')
REPORT_DIR="/tmp/claude/${SANITIZED_PATH}/reviews"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
FILEPATH="${REPORT_DIR}/code-review-${TIMESTAMP}.md"
```

Initialize with this schema:

```markdown
# Code Review Report

**Date**: {current date}
**Reviewer**: Code Review Agent
**Source**: {e.g., "PR diff", "unstaged changes", "main..HEAD"}
**Files Changed**: {count}
**Domains Detected**: {list}
**Status**: In Progress

## Summary

| Category | Items Checked | Passed | Failed | Findings |
|----------|---------------|--------|--------|----------|
| Security | 0 | 0 | 0 | 0 |
| Code Quality | 0 | 0 | 0 | 0 |
| LLM Code Smells | 0 | 0 | 0 | 0 |
| Impact Analysis | 0 | 0 | 0 | 0 |
| Simplification | 0 | 0 | 0 | 0 |
{domain-specific categories added based on detected domains}

## Detailed Findings

{findings added here as review progresses}
```

### Step 3: Review Changed Code

For EACH checklist item:

1. **Scope feedback to diff lines only** - Only flag issues in the changed code
2. **Use file context** - Read full file content to understand surrounding code
3. **Apply relevant checks** - Use domain-appropriate checklist items
4. **Document findings** - Record each violation found in changed code

**Key principle**: The diff is what gets reviewed. The rest of the file provides context to make that review accurate.

### Step 4: Impact Analysis

Check if the diff might affect other parts of the codebase:

- **Changed exports/interfaces** - Search for usages elsewhere that may break
- **Modified API signatures** - Check for callers that need updating
- **Altered shared utilities** - Look for consumers that may be affected
- **Config/schema changes** - Find code that depends on old structure

Report any unaccounted-for impacts as findings with severity based on risk.

### Step 5: Document Each Finding

For each issue found, add an entry:

```markdown
### [{CATEGORY}] {Item Name}

**Status**: FAIL
**Severity**: Critical | High | Medium | Low
**Scope**: Changed code | Impact analysis

#### Violation

- **File**: `path/to/file.ext`
- **Line(s)**: 42-48 (from diff)
- **Code**:
```
// problematic code snippet from diff
```
- **Issue**: {explanation of what's wrong}
- **Recommendation**: {how to fix it}
```

### Step 6: Finalize the Report

After completing all checks:

1. Update the summary table with final counts
2. Add an executive summary:
   - Total issues found
   - Critical issues requiring immediate attention
   - Impact analysis results
   - Recommended priority order for fixes
3. Update status to "Complete"
4. Inform the user of the report location

---

## Generic Review Categories

These checks apply to ALL changed code regardless of domain.

---

## Category: Security Fundamentals

### Hardcoded Secrets

**Check diff for**:
- API keys, passwords, secrets in changed code
- Patterns: `api_key`, `apiKey`, `password`, `secret`, `token`, `credential` with literal values

**Pass criteria**: No hardcoded secrets in diff (should use environment variables)
**Severity**: Critical

---

### Eval and Dynamic Code Execution

**Check diff for**:
- JavaScript/TypeScript: `eval(`, `new Function(`, `setTimeout("`, `setInterval("`
- Python: `eval(`, `exec(`, `compile(`

**Pass criteria**: No dynamic code execution in changed lines
**Severity**: Critical

---

### Command Injection

**Check diff for**:
- Python: `subprocess` with `shell=True`, `os.system(`
- Go: `exec.Command(` with unsanitized input

**Pass criteria**: No unvalidated user input in shell commands
**Severity**: Critical

---

## Category: Code Quality

### Console/Print Statements

**Check diff for**:
- JavaScript/TypeScript: `console.log`, `console.debug`, `console.info`
- Python: `print(` statements

**Pass criteria**: No debug statements in production code changes
**Severity**: Low

---

### TODO/FIXME Comments

**Check diff for**:
- `TODO:`, `FIXME:`, `HACK:`, `XXX:` comments

**Pass criteria**: New TODOs should be tracked in issues
**Severity**: Low

---

### Empty Catch/Except Blocks

**Check diff for**:
- JavaScript/TypeScript: `catch { }` or `catch(e) { }`
- Python: `except: pass` or empty except blocks

**Pass criteria**: All error handlers log or rethrow
**Severity**: High

---

## Category: LLM Code Smells

### Placeholder Implementations

**Check diff for**:
- `TODO`, `PLACEHOLDER`, `IMPLEMENT`, `NotImplemented`
- Functions that just `return None`, `return []`, `return {}`

**Pass criteria**: No placeholder implementations in production code
**Severity**: High

---

### Overly Generic Abstractions

**Check diff for**:
- New classes/functions with names like `GenericHandler`, `BaseManager`, `AbstractFactory`
- Abstractions without clear reuse justification

**Pass criteria**: Abstractions are justified by actual reuse
**Severity**: Low

---

## Category: Impact Analysis

### Breaking Changes

**Check if diff modifies**:
- Exported functions/classes - search for imports elsewhere
- API endpoints - search for callers
- Shared types/interfaces - search for usages
- Config schemas - search for consumers

**Pass criteria**: All impacted code identified and accounted for
**Severity**: High (if unaccounted impacts found)

---

## Category: Simplification

### Duplicate Logic

**Check diff for**:
- Repeated code patterns (not just syntactic similarity)
- Copy-pasted code with minor variations
- Similar validation, transformation, or formatting logic

**Pass criteria**: No obvious duplication in changed code
**Severity**: Medium
**Suggestion**: Extract shared logic into reusable functions

---

### Unnecessary Complexity

**Check diff for**:
- Deeply nested conditionals (more than 3 levels)
- Functions doing multiple unrelated things
- Overly complex control flow

**Pass criteria**: Code is reasonably flat and focused
**Severity**: Medium
**Suggestion**: Use early returns, extract helper functions

---

### Verbose Patterns

**Check diff for**:
- Patterns that have simpler alternatives in the language
- Redundant null checks or type assertions
- Unnecessary intermediate variables

**Pass criteria**: Code uses idiomatic patterns
**Severity**: Low
**Suggestion**: Simplify using language built-ins

---

## Domain-Specific Review Items

Based on detected domains, read and apply the appropriate checklists:

- **Frontend detected** (`designer/`): Read `frontend.md` and apply those checks to changed code
- **Backend detected** (`server/`, `rag/`, `runtimes/`): Read `backend.md` and apply those checks to changed code

---

## Final Summary Template

```markdown
## Executive Summary

**Review completed**: {timestamp}
**Total findings**: {count}

### Critical Issues (Must Fix)
1. {issue 1}
2. {issue 2}

### Impact Analysis Results
- {summary of any breaking changes or unaccounted impacts}

### High Priority (Should Fix)
1. {issue 1}
2. {issue 2}

### Recommendations
{Overall recommendations based on the changes reviewed}
```

---

## Notes for the Agent

1. **Scope to diff**: Only flag issues in the changed lines. Don't review unchanged code.

2. **Use context**: Read full files to understand the changes, but feedback targets the diff only.

3. **Check impacts**: When changes touch exports, APIs, or shared code, search for affected consumers.

4. **Be specific**: Include file paths, line numbers (from diff), and code snippets for every finding.

5. **Prioritize**: Flag critical security issues immediately.

6. **Provide solutions**: Each finding should include a recommendation for how to fix it.

7. **Update incrementally**: Update the review document after each category, not at the end.
