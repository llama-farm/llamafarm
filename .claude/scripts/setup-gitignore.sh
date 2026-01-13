#!/bin/bash
#
# Setup .gitignore for Claude Orchestration
#
# Adds .claude/ to .gitignore while ensuring Claude Code can still find it.
#
# IMPORTANT: Claude Code has NO issues finding .claude when it's in .gitignore.
# The .gitignore only affects git, not Claude's file reading capabilities.
#
# Usage:
#   ./setup-gitignore.sh                    # Setup in current project
#   ./setup-gitignore.sh /path/to/project   # Setup in specific project
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="${1:-$(pwd)}"
GITIGNORE="$PROJECT_DIR/.gitignore"

echo -e "${CYAN}Setting up .gitignore for Claude Orchestration...${NC}"
echo ""

# Create .gitignore if it doesn't exist
if [ ! -f "$GITIGNORE" ]; then
    touch "$GITIGNORE"
    echo -e "${GREEN}✓${NC} Created .gitignore"
fi

# Check if .claude is already in .gitignore
if grep -q "^\.claude/?$" "$GITIGNORE" 2>/dev/null || grep -q "^\.claude$" "$GITIGNORE" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} .claude already in .gitignore"
else
    # Add .claude to .gitignore with a comment
    echo "" >> "$GITIGNORE"
    echo "# Claude Code Orchestration System (not committed, install via install-claude-orchestration.sh)" >> "$GITIGNORE"
    echo ".claude/" >> "$GITIGNORE"
    echo -e "${GREEN}✓${NC} Added .claude/ to .gitignore"
fi

# Also ignore context files that might be created
if ! grep -q "\.claude/context/" "$GITIGNORE" 2>/dev/null; then
    # Already covered by .claude/
    :
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} Claude Code can still read .claude/ even though it's in .gitignore."
echo -e "      The .gitignore only affects git operations, not file access."
echo ""
echo -e "To export tests/demos for committing, run:"
echo -e "  ${CYAN}bash .claude/scripts/export-to-project.sh${NC}"
