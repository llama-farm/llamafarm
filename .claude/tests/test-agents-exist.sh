#!/bin/bash
#
# Test: Verify all agent files exist and have required frontmatter
#

set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
AGENTS_DIR="$PROJECT_DIR/.claude/agents"

echo "Testing agent files..."

# Required agents
AGENTS=(
    "task-orchestrator.md"
    "test-runner.md"
    "demo-builder.md"
    "debugger.md"
    "code-reviewer.md"
    "web-researcher.md"
    "smart-committer.md"
    "llamafarm.md"
    "database-architect.md"
    "api-designer.md"
    "security-auditor.md"
)

FAILED=0

for agent in "${AGENTS[@]}"; do
    if [ -f "$AGENTS_DIR/$agent" ]; then
        # Check for required frontmatter
        if grep -q "^---" "$AGENTS_DIR/$agent" && grep -q "^name:" "$AGENTS_DIR/$agent" && grep -q "^description:" "$AGENTS_DIR/$agent"; then
            echo "✓ $agent exists with valid frontmatter"
        else
            echo "✗ $agent exists but missing required frontmatter (name, description)"
            FAILED=1
        fi
    else
        echo "✗ $agent does NOT exist"
        FAILED=1
    fi
done

exit $FAILED
