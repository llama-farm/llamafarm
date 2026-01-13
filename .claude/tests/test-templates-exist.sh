#!/bin/bash
#
# Test: Verify all project templates exist
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
TEMPLATES_DIR="$PROJECT_DIR/.claude/templates"

echo "Testing project templates..."

# Required templates
TEMPLATES=(
    "new-feature.md"
    "bug-fix.md"
    "llamafarm-app.md"
    "llamafarm-contrib.md"
)

FAILED=0

for template in "${TEMPLATES[@]}"; do
    if [ -f "$TEMPLATES_DIR/$template" ]; then
        # Check for minimum content
        if [ $(wc -l < "$TEMPLATES_DIR/$template") -gt 10 ]; then
            echo "✓ $template exists with content"
        else
            echo "✗ $template exists but seems empty"
            FAILED=1
        fi
    else
        echo "✗ $template does NOT exist"
        FAILED=1
    fi
done

exit $FAILED
