#!/bin/bash
#
# Test: Verify all hook files exist and are executable
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
HOOKS_DIR="$PROJECT_DIR/.claude/hooks"

echo "Testing hook files..."

# Required hooks
HOOKS=(
    "stop-check.py"
    "pre-compact-save.py"
    "session-start.py"
    "notify-slack.sh"
    "notify-webhook.sh"
    "post-edit-lint.sh"
    "phase-commit.sh"
)

FAILED=0

for hook in "${HOOKS[@]}"; do
    if [ -f "$HOOKS_DIR/$hook" ]; then
        if [ -x "$HOOKS_DIR/$hook" ]; then
            echo "✓ $hook exists and is executable"
        else
            echo "✗ $hook exists but is NOT executable"
            FAILED=1
        fi
    else
        echo "✗ $hook does NOT exist"
        FAILED=1
    fi
done

exit $FAILED
