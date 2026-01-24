#!/bin/bash
#
# Test: Verify all utility scripts exist and are executable
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SCRIPTS_DIR="$PROJECT_DIR/.claude/scripts"

echo "Testing utility scripts..."

# Required scripts
SCRIPTS=(
    "run-headless.sh"
    "slack-handler.sh"
    "run-all-tests.sh"
    "run-all-demos.sh"
)

FAILED=0

for script in "${SCRIPTS[@]}"; do
    if [ -f "$SCRIPTS_DIR/$script" ]; then
        if [ -x "$SCRIPTS_DIR/$script" ]; then
            echo "✓ $script exists and is executable"
        else
            echo "✗ $script exists but is NOT executable"
            FAILED=1
        fi
    else
        echo "✗ $script does NOT exist"
        FAILED=1
    fi
done

exit $FAILED
