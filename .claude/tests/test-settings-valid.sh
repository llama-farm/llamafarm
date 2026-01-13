#!/bin/bash
#
# Test: Verify settings.json is valid JSON and has required hooks
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SETTINGS_FILE="$PROJECT_DIR/.claude/settings.json"

echo "Testing settings.json..."

# Check file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "✗ settings.json does NOT exist"
    exit 1
fi

# Check valid JSON
if ! python3 -c "import json; json.load(open('$SETTINGS_FILE'))" 2>/dev/null; then
    echo "✗ settings.json is NOT valid JSON"
    exit 1
fi
echo "✓ settings.json is valid JSON"

# Check required hooks
REQUIRED_HOOKS=("Stop" "PreCompact" "SessionStart" "Notification" "PostToolUse")

for hook in "${REQUIRED_HOOKS[@]}"; do
    if python3 -c "
import json
d = json.load(open('$SETTINGS_FILE'))
assert '$hook' in d.get('hooks', {}), 'Missing $hook'
" 2>/dev/null; then
        echo "✓ $hook hook is configured"
    else
        echo "✗ $hook hook is MISSING"
        exit 1
    fi
done

echo "✓ All required hooks are configured"
exit 0
