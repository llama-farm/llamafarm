#!/bin/bash
#
# Test: Verify LlamaFarm reference documentation exists and has content
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
DOCS_DIR="$PROJECT_DIR/.claude/docs"
REF_FILE="$DOCS_DIR/LLAMAFARM-REFERENCE.md"

echo "Testing LlamaFarm reference..."

# Check file exists
if [ ! -f "$REF_FILE" ]; then
    echo "✗ LLAMAFARM-REFERENCE.md does NOT exist"
    exit 1
fi
echo "✓ LLAMAFARM-REFERENCE.md exists"

# Check minimum content (should be substantial)
LINE_COUNT=$(wc -l < "$REF_FILE")
if [ "$LINE_COUNT" -lt 100 ]; then
    echo "✗ LLAMAFARM-REFERENCE.md seems too short ($LINE_COUNT lines)"
    exit 1
fi
echo "✓ LLAMAFARM-REFERENCE.md has substantial content ($LINE_COUNT lines)"

# Check for key sections
REQUIRED_SECTIONS=(
    "Documentation Links"
    "Quick Start"
    "llamafarm.yaml"
    "API"
    "Anomaly Detection"
    "RAG"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
    if grep -qi "$section" "$REF_FILE"; then
        echo "✓ Contains section: $section"
    else
        echo "✗ Missing section: $section"
        exit 1
    fi
done

exit 0
