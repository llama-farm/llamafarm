#!/bin/bash
#
# Run All Demos - Master demo runner for the Claude orchestration system
#
# Usage: ./run-all-demos.sh [demo-name]
#
# Runs all demo scripts in .claude/demos/ or a specific demo if named.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
DEMOS_DIR="$PROJECT_DIR/.claude/demos"
CONTEXT_DIR="$PROJECT_DIR/.claude/context"

# Ensure directories exist
mkdir -p "$CONTEXT_DIR"

# Check for specific demo
SPECIFIC_DEMO="$1"

echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}  Demo Runner${NC}"
echo -e "${CYAN}================================${NC}"
echo ""

# Track results
EXECUTED=()
PENDING=()

run_demo() {
    local demo_file="$1"
    local demo_name=$(basename "$demo_file")

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Demo: $demo_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    if bash "$demo_file"; then
        EXECUTED+=("$demo_name")
        echo ""
        echo -e "${GREEN}✓ Demo completed successfully${NC}"
    else
        echo ""
        echo -e "${RED}✗ Demo failed${NC}"
        return 1
    fi
    echo ""
}

# Find and run demos
if [ -d "$DEMOS_DIR" ]; then
    if [ -n "$SPECIFIC_DEMO" ]; then
        # Run specific demo
        demo_file="$DEMOS_DIR/$SPECIFIC_DEMO"
        if [ ! -f "$demo_file" ]; then
            demo_file="$DEMOS_DIR/demo-$SPECIFIC_DEMO.sh"
        fi
        if [ -f "$demo_file" ]; then
            run_demo "$demo_file"
        else
            echo -e "${RED}Demo not found: $SPECIFIC_DEMO${NC}"
            echo ""
            echo "Available demos:"
            ls -1 "$DEMOS_DIR"/*.sh 2>/dev/null | xargs -n1 basename
            exit 1
        fi
    else
        # Run all demos
        for demo_file in "$DEMOS_DIR"/demo-*.sh; do
            if [ -f "$demo_file" ]; then
                run_demo "$demo_file" || true
                echo ""
                echo -e "${YELLOW}Press Enter to continue to next demo (or Ctrl+C to stop)...${NC}"
                read -r
            fi
        done
    fi
else
    echo -e "${YELLOW}No demos directory found at $DEMOS_DIR${NC}"
    echo "Creating demos directory..."
    mkdir -p "$DEMOS_DIR"
fi

# Summary
echo -e "${CYAN}================================${NC}"
echo -e "${CYAN}  Demo Summary${NC}"
echo -e "${CYAN}================================${NC}"
echo ""
echo "Executed: ${#EXECUTED[@]} demos"
for demo in "${EXECUTED[@]}"; do
    echo -e "  ${GREEN}✓${NC} $demo"
done

# Save results to JSON
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$CONTEXT_DIR/demo-results.json" << EOF
{
  "last_run": "$TIMESTAMP",
  "executed": $(printf '%s\n' "${EXECUTED[@]}" | jq -R . | jq -s .),
  "pending": $(printf '%s\n' "${PENDING[@]}" | jq -R . | jq -s .)
}
EOF

echo ""
echo "Results saved to $CONTEXT_DIR/demo-results.json"
