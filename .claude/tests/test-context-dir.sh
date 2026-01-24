#!/bin/bash
#
# Test: Verify context directory and files are set up correctly
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CONTEXT_DIR="$PROJECT_DIR/.claude/context"

echo "Testing context directory..."

# Check directory exists
if [ ! -d "$CONTEXT_DIR" ]; then
    echo "✗ context directory does NOT exist"
    exit 1
fi
echo "✓ context directory exists"

# Check progress.json exists and is valid
if [ -f "$CONTEXT_DIR/progress.json" ]; then
    if python3 -c "import json; json.load(open('$CONTEXT_DIR/progress.json'))" 2>/dev/null; then
        echo "✓ progress.json is valid JSON"
    else
        echo "✗ progress.json is NOT valid JSON"
        exit 1
    fi
else
    echo "✗ progress.json does NOT exist"
    exit 1
fi

# Check progress.json has required fields
python3 -c "
import json
d = json.load(open('$CONTEXT_DIR/progress.json'))
required = ['tasks', 'tests', 'demos']
for field in required:
    assert field in d, f'Missing field: {field}'
print('✓ progress.json has required fields')
"

exit 0
