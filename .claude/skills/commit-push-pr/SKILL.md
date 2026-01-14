---
name: commit-push-pr
description: Commit changes, push to GitHub, and open a PR. Includes quality checks (security, patterns, simplification). Use --quick to skip checks.
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, AskUserQuestion
---

# Commit, Push & PR Skill

Automates the git workflow of committing changes, pushing to GitHub, and opening a PR with intelligent handling of edge cases.

## Required Reading

Before executing, internalize the git workflow standards:
@.claude/rules/git_workflow.md

Key rules:
- Use Conventional Commits format: `type(scope): description`
- **NEVER attribute Claude** in commits or PRs (no co-author, no mentions)
- **NEVER skip pre-commit hooks** (no `--no-verify`)

---

## Execution Workflow

### Step 1: Assess Git State

Run these commands to understand the current state:

```bash
# Detect the default branch (main, master, etc.)
DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')
# Fallback if symbolic-ref fails (e.g., shallow clone or missing HEAD)
if [ -z "$DEFAULT_BRANCH" ]; then
  DEFAULT_BRANCH=$(git remote show origin 2>/dev/null | grep 'HEAD branch' | awk '{print $NF}')
fi
# Final fallback to 'main' if detection fails
DEFAULT_BRANCH=${DEFAULT_BRANCH:-main}

# Get current branch
BRANCH=$(git branch --show-current)

# Check for uncommitted changes
git status --porcelain

# Check for unpushed commits (if branch has upstream)
git log origin/$DEFAULT_BRANCH..HEAD --oneline 2>/dev/null || echo "No upstream or no commits ahead"

# Check if branch has upstream tracking
git rev-parse --abbrev-ref @{upstream} 2>/dev/null || echo "No upstream"
```

Determine the state:
- `HAS_CHANGES`: Are there uncommitted changes (staged, unstaged, or untracked)?
- `HAS_UNPUSHED`: Are there commits ahead of origin/$DEFAULT_BRANCH?
- `ON_DEFAULT_BRANCH`: Is current branch the default branch ($DEFAULT_BRANCH)?
- `HAS_UPSTREAM`: Does the branch track a remote?

### Step 2: Handle "Nothing to Do" Case

If `!HAS_CHANGES && !HAS_UNPUSHED`:
```
Inform user: "No changes to commit and no unpushed commits. Nothing to do."
Exit gracefully.
```

### Step 3: Handle "No Changes But Unpushed Commits" Case

If `!HAS_CHANGES && HAS_UNPUSHED`:

1. Check if PR already exists:
```bash
gh pr list --head "$(git branch --show-current)" --json number,url,title
```

2. If PR exists:
   - Offer to push updates to the existing PR
   - Report the PR URL

3. If no PR:
   - Offer to push and create a new PR
   - Proceed to Step 7

### Step 4: Branch Management (if HAS_CHANGES)

**If on default branch ($DEFAULT_BRANCH):**

1. Inform user that changes need to go on a feature branch
2. Stage changes first to analyze them:
```bash
git add -A
git diff --staged --stat
```

3. Generate a conventional commit message based on the changes (see Step 5)

4. Derive branch name from commit message:
   - `feat(cli): add project list` → `feat-cli-add-project-list`
   - `fix: resolve memory leak` → `fix-resolve-memory-leak`
   - Rules: lowercase, replace spaces/special chars with hyphens, max 50 chars

5. Create and checkout the new branch:
```bash
git checkout -b <branch-name>
```

**If already on feature branch:**
- Continue with the existing branch
- Check if PR exists for context

### Step 5: Stage Changes and Generate Commit Message

1. Stage all changes:
```bash
git add -A
```

2. Analyze the staged changes:
```bash
git diff --staged --stat
git diff --staged
```

3. Generate a conventional commit message based on:
   - Files changed (infer scope from directory)
   - Nature of changes (feat/fix/refactor/docs/test/chore)
   - Summarize the "why" not just the "what"

4. Present the commit message to the user. Example format:
```
Proposed commit message:

  feat(cli): add project listing command

  Adds a new 'lf project list' command that displays all projects
  in the current workspace with their status.

Do you want to use this message, modify it, or provide your own?
```

### Step 5.5: Quality Check

**Skip if**: `--quick` flag was passed.

Run quality checks on staged changes before committing.

#### 1. Auto-fix trivial issues (no prompt needed)

Search for and remove debug statements:

```bash
# Find files with debug statements
git diff --staged --name-only | xargs grep -l -E "(console\.(log|debug|info)|debugger|print\()" 2>/dev/null
```

For each file found:
- Remove `console.log(...)`, `console.debug(...)`, `console.info(...)` statements
- Remove `debugger;` statements
- Remove `print(...)` statements (Python)
- Re-stage the file after fixes

Report: "Auto-fixed: Removed N debug statements from M files"

#### 2. Check for issues requiring attention

Scan staged diff for:

| Issue | Severity | Action |
|-------|----------|--------|
| Hardcoded secrets (API keys, passwords) | BLOCK | Cannot auto-fix - user must remove |
| Command injection (`shell=True`, `os.system`) | BLOCK | Cannot auto-fix - user must refactor |
| Empty catch/except blocks | PROPOSE | Suggest adding error logging |
| Duplicate code patterns | PROPOSE | Suggest extraction |
| Unused imports | PROPOSE | Suggest removal |
| TODO/FIXME comments | WARN | Note but allow proceed |

#### 3. Handle blocking issues

If BLOCK issues found:
- List each issue with file:line reference
- Stop the workflow
- User must fix manually and re-run

#### 4. Handle proposable fixes

For each PROPOSE issue:
- Show: file, line, problem, suggested fix
- Ask: "Apply this fix? (y/n/all/skip)"
- If approved: apply edit, re-stage
- If skipped: continue without fix

#### 5. Handle warnings

For WARN issues:
- Display summary
- Continue without blocking

---

### Step 6: Create the Commit

Create the commit with the approved message:

```bash
git commit -m "$(cat <<'EOF'
type(scope): short description

Optional longer description explaining the change.
EOF
)"
```

**Important:**
- Use HEREDOC for multi-line messages
- Never add co-author or Claude attribution
- Let pre-commit hooks run (never use `--no-verify`)

**If commit fails due to pre-commit hook:**
- Report the failure to the user
- Show the hook output
- Do NOT retry with `--no-verify`
- Ask user how to proceed (fix issues or abort)

### Step 7: Push to Remote

1. Check if branch has upstream:
```bash
git rev-parse --abbrev-ref @{upstream} 2>/dev/null
```

2. If no upstream, push with `-u`:
```bash
git push -u origin $(git branch --show-current)
```

3. If has upstream, regular push:
```bash
git push
```

**If push fails due to conflicts:**
- Inform user about the conflict
- Suggest: `git pull --rebase origin $DEFAULT_BRANCH` or `git merge origin/$DEFAULT_BRANCH`
- Do NOT force push

### Step 8: Create or Report PR

1. Check if PR already exists:
```bash
gh pr list --head "$(git branch --show-current)" --json number,url,title
```

2. **If PR exists:**
   - Report: "Changes pushed to existing PR: <URL>"
   - Show PR title and number

3. **If no PR exists:**
   - Generate PR title from commit message (first line)
   - Generate PR body with summary of changes
   - Create PR:

```bash
gh pr create --title "type(scope): description" --body "$(cat <<'EOF'
## Summary

- Brief description of changes

## Changes

- List of key changes made

## Test Plan

- How to verify these changes work
EOF
)"
```

4. Report the new PR URL to the user

---

## Branch Name Generation

Convert commit message to valid branch name:

| Input | Output |
|-------|--------|
| `feat(cli): add project list command` | `feat-cli-add-project-list-command` |
| `fix: resolve memory leak in cache` | `fix-resolve-memory-leak-in-cache` |
| `refactor(server): simplify auth flow` | `refactor-server-simplify-auth-flow` |

Algorithm:
1. Take the commit message (first line only)
2. Lowercase everything
3. Remove the colon after type/scope
4. Replace `(` and `)` with `-`
5. Replace spaces and special characters with `-`
6. Collapse multiple hyphens to single hyphen
7. Trim to max 50 characters at word boundary
8. Remove trailing hyphens

---

## Error Handling

| Error | Action |
|-------|--------|
| Pre-commit hook fails | Show output, ask user to fix, do NOT bypass |
| Push rejected (conflicts) | Suggest rebase/merge, do NOT force push |
| PR creation fails | Show error, suggest manual creation |
| Not a git repo | Inform user, exit |
| gh CLI not installed | Inform user how to install |
| Not authenticated to GitHub | Suggest `gh auth login` |

---

## Output Format

On success, report:
```
Committed: feat(cli): add project list command
Branch: feat-cli-add-project-list-command
Pushed to: origin/feat-cli-add-project-list-command
PR: https://github.com/owner/repo/pull/123
```

---

## Notes for the Agent

1. **Never mention Claude** - No co-author lines, no "generated by Claude" in PR descriptions
2. **Respect hooks** - Pre-commit hooks exist for a reason, never skip them
3. **Be informative** - Tell the user what's happening at each step
4. **Handle errors gracefully** - Don't leave the repo in a broken state
5. **Ask when uncertain** - If the commit message isn't clear, ask the user
6. **Keep it simple** - One commit per invocation, clear linear workflow
