#!/bin/bash
#
# Phase Commit Hook
#
# Creates a git commit after phase completion.
# IMPORTANT: Never attributes to Claude - uses git config user.
#
# Usage: ./phase-commit.sh "Phase X: Description"
#
# Environment variables:
#   CLAUDE_SKIP_COMMIT - Set to "1" to skip committing
#   CLAUDE_DRY_RUN - Set to "1" to show what would be committed without committing
#

set -e

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Check if we should skip
if [ "$CLAUDE_SKIP_COMMIT" = "1" ]; then
    echo "[COMMIT] Skipping commit (CLAUDE_SKIP_COMMIT=1)"
    exit 0
fi

# Get commit message from argument or generate default
PHASE_MSG="${1:-Automated phase commit}"

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "[COMMIT] Not a git repository, skipping commit"
    exit 0
fi

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
    echo "[COMMIT] No changes to commit"
    exit 0
fi

# Run linters before commit (Python with ruff)
echo "[COMMIT] Running pre-commit linters..."

# Find and lint Python files that are staged or modified
PYTHON_FILES=$(git diff --name-only --diff-filter=ACMR HEAD 2>/dev/null | grep '\.py$' || true)
if [ -n "$PYTHON_FILES" ] && command -v ruff &> /dev/null; then
    echo "[COMMIT] Linting Python files with ruff..."
    for file in $PYTHON_FILES; do
        if [ -f "$file" ]; then
            ruff check --fix "$file" 2>/dev/null || true
            ruff format "$file" 2>/dev/null || true
        fi
    done
fi

# Stage all changes
git add -A

# Check again after linting
if git diff --cached --quiet; then
    echo "[COMMIT] No changes to commit after linting"
    exit 0
fi

# Get test results if available
TEST_INFO=""
TEST_RESULTS_FILE="$PROJECT_DIR/.claude/context/test-results.json"
if [ -f "$TEST_RESULTS_FILE" ]; then
    PASSED=$(python3 -c "import json; d=json.load(open('$TEST_RESULTS_FILE')); print(d.get('passed', 0))" 2>/dev/null || echo "0")
    FAILED=$(python3 -c "import json; d=json.load(open('$TEST_RESULTS_FILE')); print(d.get('failed', 0))" 2>/dev/null || echo "0")
    TEST_INFO="Tests: $PASSED passed, $FAILED failed"
fi

# Build commit message
COMMIT_MSG="$PHASE_MSG"
if [ -n "$TEST_INFO" ]; then
    COMMIT_MSG="$COMMIT_MSG

$TEST_INFO"
fi

# Dry run mode
if [ "$CLAUDE_DRY_RUN" = "1" ]; then
    echo "[COMMIT] DRY RUN - Would commit with message:"
    echo "---"
    echo "$COMMIT_MSG"
    echo "---"
    echo "[COMMIT] Files to be committed:"
    git diff --cached --name-only
    exit 0
fi

# Create the commit (NO Claude attribution)
git commit -m "$COMMIT_MSG"

echo "[COMMIT] Created commit: $PHASE_MSG"
git log -1 --oneline

exit 0
