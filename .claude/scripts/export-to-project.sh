#!/bin/bash
#
# Export Tests and Demos from .claude to Project Root
#
# This script copies tests and demos from .claude/ to project-level directories
# so they can be committed to version control while .claude stays in .gitignore.
#
# Usage:
#   ./export-to-project.sh                    # Export to current project
#   ./export-to-project.sh /path/to/project   # Export to specific project
#
# What it exports:
#   .claude/tests/   -> tests/claude-orchestration/
#   .claude/demos/   -> demos/claude-orchestration/
#
# The exported tests/demos are standalone and don't depend on .claude/
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Target project directory
PROJECT_DIR="${1:-$(pwd)}"
CLAUDE_DIR="$PROJECT_DIR/.claude"

echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Export Claude Orchestration Tests & Demos                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check .claude exists
if [ ! -d "$CLAUDE_DIR" ]; then
    echo -e "${RED}Error: .claude directory not found at $CLAUDE_DIR${NC}"
    exit 1
fi

# Export directories
TESTS_SRC="$CLAUDE_DIR/tests"
DEMOS_SRC="$CLAUDE_DIR/demos"
TESTS_DST="$PROJECT_DIR/tests/claude-orchestration"
DEMOS_DST="$PROJECT_DIR/demos/claude-orchestration"

# Function to export a directory
export_dir() {
    local src="$1"
    local dst="$2"
    local name="$3"

    if [ ! -d "$src" ]; then
        echo -e "${YELLOW}Skipping $name: source not found${NC}"
        return
    fi

    # Count files
    local count=$(find "$src" -type f -name "*.sh" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$count" -eq 0 ]; then
        echo -e "${YELLOW}Skipping $name: no files to export${NC}"
        return
    fi

    echo -e "${YELLOW}Exporting $name ($count files)...${NC}"

    # Create destination
    mkdir -p "$dst"

    # Copy files (excluding any that reference internal .claude paths)
    for file in "$src"/*.sh; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")

            # Copy and make standalone (update paths)
            sed -e 's|\$CLAUDE_PROJECT_DIR/.claude|$(dirname "\$0")/../../.claude|g' \
                -e 's|PROJECT_DIR:-\$(pwd)|PROJECT_DIR:-$(cd "$(dirname "$0")/../.." \&\& pwd)|g' \
                "$file" > "$dst/$filename"

            chmod +x "$dst/$filename"
            echo -e "  ${GREEN}✓${NC} $filename"
        fi
    done
}

# Export tests
export_dir "$TESTS_SRC" "$TESTS_DST" "tests"

# Export demos
export_dir "$DEMOS_SRC" "$DEMOS_DST" "demos"

# Create a README in each exported directory
create_readme() {
    local dst="$1"
    local type="$2"

    if [ -d "$dst" ]; then
        cat > "$dst/README.md" << EOF
# Claude Orchestration $type

These $type were exported from the \`.claude/\` directory and are standalone.

## Running

\`\`\`bash
# Run all $type
for f in *.sh; do bash "\$f"; done

# Run a specific ${type%s}
bash ${type%s}-name.sh
\`\`\`

## Note

These are copies exported for version control. The original source is in \`.claude/${type}/\`.
To update, re-run: \`bash .claude/scripts/export-to-project.sh\`
EOF
        echo -e "  ${GREEN}✓${NC} Created README.md"
    fi
}

echo ""
echo -e "${YELLOW}Creating README files...${NC}"
create_readme "$TESTS_DST" "tests"
create_readme "$DEMOS_DST" "demos"

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Export Complete!                                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Exported to:"
[ -d "$TESTS_DST" ] && echo -e "  ${BLUE}$TESTS_DST${NC}"
[ -d "$DEMOS_DST" ] && echo -e "  ${BLUE}$DEMOS_DST${NC}"
echo ""
echo -e "These directories can be committed to git."
echo -e "Remember to add ${YELLOW}.claude/${NC} to ${YELLOW}.gitignore${NC}"
