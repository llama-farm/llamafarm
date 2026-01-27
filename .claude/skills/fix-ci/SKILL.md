---
name: fix-ci
description: Fetch GitHub CI failure information, analyze root causes, reproduce locally, and propose a fix plan. Use `/fix-ci` for current branch or `/fix-ci <run-id>` for a specific run.
allowed-tools: Bash, Read, Grep, Glob, Task, AskUserQuestion, EnterPlanMode
---

# Fix CI Skill

Automates CI troubleshooting by fetching GitHub Actions failures, analyzing logs, reproducing issues locally, and creating a fix plan for user approval.

---

## Execution Workflow

### Step 1: Prerequisites Check

Verify the GitHub CLI is installed and authenticated:

```bash
gh --version && gh auth status
```

**If gh is not installed:**
- Inform user: "GitHub CLI is required. Install with: `brew install gh`"
- Exit gracefully

**If not authenticated:**
- Inform user: "Please authenticate with: `gh auth login`"
- Exit gracefully

### Step 2: Parse Arguments

Determine the mode based on arguments:

- **No arguments** (`/fix-ci`): Fetch failures for the current branch only
- **With run-id** (`/fix-ci <run-id>`): Fetch specific run (bypasses branch scoping)

### Step 3: Fetch Failed Run

**Default mode (current branch):**

```bash
BRANCH=$(git branch --show-current)
gh run list --branch "$BRANCH" --status failure --limit 1 --json databaseId,name,headBranch,workflowName,createdAt
```

**Specific run mode:**

```bash
gh run view <run-id> --json databaseId,name,headBranch,workflowName,jobs,conclusion
```

**If no failures found:**
- Report: "No failed runs found for branch `$BRANCH`. CI is green!"
- Optionally show recent successful runs:
```bash
gh run list --branch "$BRANCH" --limit 3 --json databaseId,conclusion,workflowName,createdAt
```
- Exit gracefully

### Step 4: Get Failure Details

Once a failed run is identified, gather comprehensive details:

```bash
RUN_ID=<the-run-id>

# Get failed jobs with their steps
gh run view $RUN_ID --json jobs --jq '.jobs[] | select(.conclusion == "failure") | {name, conclusion, steps: [.steps[] | select(.conclusion == "failure")]}'

# Get failed step logs (critical for debugging)
gh run view $RUN_ID --log-failed 2>&1 | head -500

# Get verbose run info
gh run view $RUN_ID --verbose
```

**Log handling:**
- Truncate logs to 500 lines to avoid context overflow
- Note to user: "Showing first 500 lines of failed logs. Full logs available on GitHub."

### Step 5: Download Artifacts (if available)

Attempt to download any debug artifacts:

```bash
# Try common artifact names - failures are OK (not all runs have artifacts)
gh run download $RUN_ID -n "coverage" -D /tmp/ci-debug/ 2>/dev/null || true
gh run download $RUN_ID -n "test-results" -D /tmp/ci-debug/ 2>/dev/null || true
gh run download $RUN_ID -n "logs" -D /tmp/ci-debug/ 2>/dev/null || true
```

If artifacts downloaded, read them for additional context.

### Step 6: Analyze Failure Type

Categorize the failure based on log patterns:

| Pattern | Failure Type | Root Cause Area |
|---------|--------------|-----------------|
| `FAIL:`, `--- FAIL`, `FAILED` | Test Failure | Specific test case |
| `ruff check`, `ruff format` | Lint Error | Code style/formatting |
| `ModuleNotFoundError`, `ImportError` | Import Error | Missing dependency |
| `TypeError`, `AttributeError` | Runtime Error | Type mismatch |
| `SyntaxError` | Syntax Error | Invalid code |
| `AssertionError` | Assertion Failure | Test expectation mismatch |
| `TimeoutError`, `timed out` | Timeout | Performance/hang |
| `PermissionError`, `EACCES` | Permission Error | File/resource access |
| `ConnectionError`, `ECONNREFUSED` | Network Error | External service |

Extract key information:
- Failed test name/file (if applicable)
- Error message
- Stack trace location (file:line)
- Environment variables or config issues

### Step 7: Map to Local Test Commands

Determine the appropriate local command based on the CI job:

| CI Workflow/Job | Local Command |
|-----------------|---------------|
| `test-cli` | `cd cli && go test ./...` |
| `test-python` (server) | `cd server && uv run pytest -v` |
| `test-python` (rag) | `cd rag && uv run pytest -v` |
| `test-python` (config) | `cd config && uv run pytest -v` |
| `test-python` (runtime) | `cd runtimes/universal && uv run pytest -v` |
| `lint` (python) | `uv run ruff check .` |
| `lint` (go) | `cd cli && golangci-lint run` |
| `type-check` | `uv run mypy .` |
| `build-cli` | `nx build cli` |
| `build-designer` | `cd designer && npm run build` |

**For specific test failures**, narrow down the command:
- Python: `cd <dir> && uv run pytest -v <test_file>::<test_name>`
- Go: `cd cli && go test -v -run <TestName> ./...`

### Step 8: Reproduce Locally

Run the mapped local command to confirm the failure reproduces:

```bash
# Example for Python test
cd server && uv run pytest -v tests/test_api.py::test_health_check
```

**Outcome A - Failure reproduces locally:**
- Good! Continue to fix plan
- Report: "Successfully reproduced failure locally"

**Outcome B - Failure does NOT reproduce locally:**
- Note: "Could not reproduce locally. Possible causes:"
  - Flaky test (timing-dependent)
  - Environment difference (CI has different deps/config)
  - Race condition
- Suggest: "Consider re-running CI with `gh run rerun $RUN_ID`"
- Ask user how to proceed (investigate further or skip)

### Step 9: Analyze Root Cause

Based on the failure type and logs, identify:

1. **What failed**: Specific test, lint rule, or build step
2. **Why it failed**: The actual error condition
3. **Where to fix**: File(s) and line(s) that need changes
4. **How to fix**: Proposed changes

Use available tools to explore:
- Read the failing test file
- Read the code being tested
- Search for related patterns in the codebase
- Check recent changes that might have caused the failure

### Step 10: Enter Plan Mode

Use `EnterPlanMode` to create a formal fix plan. The plan should include:

```markdown
# CI Fix Plan

## Problem Statement
[Summary of the CI failure from logs]

## Failure Details
- **Run ID**: <run-id>
- **Workflow**: <workflow-name>
- **Job**: <job-name>
- **Error Type**: <categorized-type>

## Root Cause Analysis
[Explanation of why the failure occurred]

## Affected Files
- `path/to/file1.py` (line X)
- `path/to/file2.py` (line Y)

## Proposed Changes

### Change 1: [Brief description]
[Specific edit to make]

### Change 2: [Brief description]
[Specific edit to make]

## Verification Steps
1. Run: `<local-test-command>`
2. Expected: All tests pass
3. Optional: Run full test suite with `<full-suite-command>`

## Notes
- [Any caveats or considerations]
```

### Step 11: User Approval Gate

Present the plan and wait for explicit user approval:
- User approves: Proceed to execute fixes
- User modifies: Incorporate feedback, update plan
- User rejects: Exit gracefully without changes

**CRITICAL**: Never make code changes without user approval.

### Step 12: Execute Fix (after approval only)

1. Make the proposed code changes using Edit tool
2. Run local tests to verify the fix:
```bash
<local-test-command>
```
3. Report results:
   - Success: "Fix verified locally. Tests pass."
   - Failure: "Fix did not resolve the issue. [details]"

**IMPORTANT**: Do NOT auto-commit changes. Leave committing to the user or `/commit-push-pr` skill.

---

## Error Handling

| Scenario | Action |
|----------|--------|
| gh CLI not installed | Direct user to install: `brew install gh` |
| gh not authenticated | Direct user to: `gh auth login` |
| No failures found | Report CI is green, exit gracefully |
| Rate limit exceeded | Suggest waiting or using `gh auth refresh` |
| Run not found | Verify run ID, suggest `gh run list` to find valid IDs |
| Large logs (>500 lines) | Truncate, note full logs on GitHub |
| Local reproduction fails | Note as flaky/env issue, offer re-run option |
| Network errors | Suggest retry, check connection |

---

## Output Format

**On finding a failure:**
```
CI Failure Found
Run: #12345 (workflow-name)
Branch: feature-branch
Failed Job: test-python
Error Type: Test Failure

Analyzing logs...
[Summary of failure]

Reproducing locally...
[Result]

Entering plan mode to propose fix...
```

**On success (after fix):**
```
Fix Applied
- Modified: path/to/file.py
- Verification: Tests pass locally

Next steps:
- Review the changes
- Run `/commit-push-pr` to commit and push
- CI will re-run automatically on push
```

---

## Notes for the Agent

1. **Always scope to current branch by default** - Users expect `/fix-ci` to fix their current work, not random failures
2. **Truncate logs wisely** - CI logs can be huge; extract the relevant error sections
3. **Reproduce before fixing** - Don't propose fixes for issues that can't be reproduced
4. **Plan mode is mandatory** - Always use EnterPlanMode before making changes
5. **Never auto-commit** - The user controls when changes are committed
6. **Be specific in analysis** - Generic advice isn't helpful; identify exact files and lines
7. **Handle flaky tests** - If reproduction fails, acknowledge it might be flaky
