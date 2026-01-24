#!/bin/bash
#
# Run All Tests - Master test runner for the Claude orchestration system
#
# Usage: ./run-all-tests.sh
#
# Runs all test scripts in .claude/tests/ and reports results.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
TESTS_DIR="$PROJECT_DIR/.claude/tests"
CONTEXT_DIR="$PROJECT_DIR/.claude/context"

# Ensure directories exist
mkdir -p "$CONTEXT_DIR"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Running All Tests${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Track results
PASSED=0
FAILED=0
SKIPPED=0
FAILURES=()

# Find and run all test scripts
if [ -d "$TESTS_DIR" ]; then
    for test_file in "$TESTS_DIR"/test-*.sh; do
        if [ -f "$test_file" ]; then
            test_name=$(basename "$test_file")
            echo -e "${YELLOW}Running: $test_name${NC}"

            if bash "$test_file" > /tmp/test_output.txt 2>&1; then
                echo -e "${GREEN}✓ PASSED${NC}"
                ((PASSED++))
            else
                echo -e "${RED}✗ FAILED${NC}"
                cat /tmp/test_output.txt
                ((FAILED++))
                FAILURES+=("$test_name")
            fi
            echo ""
        fi
    done
else
    echo -e "${YELLOW}No tests directory found at $TESTS_DIR${NC}"
    echo "Creating tests directory..."
    mkdir -p "$TESTS_DIR"
fi

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  Test Results${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo ""

# Save results to JSON
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$CONTEXT_DIR/test-results.json" << EOF
{
  "last_run": "$TIMESTAMP",
  "passed": $PASSED,
  "failed": $FAILED,
  "skipped": $SKIPPED,
  "failures": $(printf '%s\n' "${FAILURES[@]}" | jq -R . | jq -s .)
}
EOF

echo "Results saved to $CONTEXT_DIR/test-results.json"

# Update progress.json with test results
python3 - << PYTHON
import json
from pathlib import Path
from datetime import datetime

context_dir = Path("$CONTEXT_DIR")
progress_file = context_dir / "progress.json"

# Load or initialize progress.json
if progress_file.exists():
    try:
        with open(progress_file) as f:
            progress = json.load(f)
    except:
        progress = {}
else:
    progress = {}

# Update test results
progress["tests"] = {
    "last_run": "$TIMESTAMP",
    "passed": $PASSED,
    "failed": $FAILED,
    "skipped": $SKIPPED
}

# Initialize other fields if missing
if not progress.get("session_id"):
    progress["session_id"] = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if not progress.get("started_at"):
    progress["started_at"] = datetime.now().isoformat()
if "tasks" not in progress:
    progress["tasks"] = {"total": 0, "completed": 0, "in_progress": 0, "pending": 0}
if "demos" not in progress:
    progress["demos"] = {"executed": [], "pending": []}
progress["iterations"] = progress.get("iterations", 0) + 1

with open(progress_file, "w") as f:
    json.dump(progress, f, indent=2)

print("Updated progress.json with test results")
PYTHON

# Exit with failure if any tests failed
if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
