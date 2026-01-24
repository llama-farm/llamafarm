#!/bin/bash
#
# Run Headless - Wrapper for running Claude Code headlessly
#
# Usage:
#   ./run-headless.sh "Your prompt here"
#   ./run-headless.sh "Your prompt" /path/to/project
#   ./run-headless.sh --template new-feature "Build a user API"
#
# Environment variables:
#   CLAUDE_ALLOWED_TOOLS - Override default allowed tools
#   CLAUDE_OUTPUT_FORMAT - json (default) or text
#   CLAUDE_SESSION_ID - Resume a specific session
#

set -e

# Parse arguments
TEMPLATE=""
PROMPT=""
PROJECT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --template|-t)
            TEMPLATE="$2"
            shift 2
            ;;
        --project|-p)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --session|-s)
            CLAUDE_SESSION_ID="$2"
            shift 2
            ;;
        *)
            if [ -z "$PROMPT" ]; then
                PROMPT="$1"
            elif [ -z "$PROJECT_DIR" ]; then
                PROJECT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Default project directory
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
export CLAUDE_PROJECT_DIR="$PROJECT_DIR"

# Ensure .claude directory exists
mkdir -p "$PROJECT_DIR/.claude/context"

# Default allowed tools
ALLOWED_TOOLS="${CLAUDE_ALLOWED_TOOLS:-Bash,Read,Edit,Write,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite}"

# Default output format
OUTPUT_FORMAT="${CLAUDE_OUTPUT_FORMAT:-json}"

# Load template if specified
if [ -n "$TEMPLATE" ]; then
    TEMPLATE_FILE="$PROJECT_DIR/.claude/templates/${TEMPLATE}.md"
    if [ -f "$TEMPLATE_FILE" ]; then
        TEMPLATE_CONTENT=$(cat "$TEMPLATE_FILE")
        PROMPT="Follow this template:\n\n$TEMPLATE_CONTENT\n\nTask: $PROMPT"
    else
        echo "Warning: Template '$TEMPLATE' not found at $TEMPLATE_FILE" >&2
    fi
fi

# Check for prompt
if [ -z "$PROMPT" ]; then
    echo "Usage: ./run-headless.sh [--template name] \"Your prompt here\" [/path/to/project]"
    exit 1
fi

# Build claude command
CLAUDE_CMD="claude -p \"$PROMPT\" --allowedTools \"$ALLOWED_TOOLS\" --output-format $OUTPUT_FORMAT"

# Add session resume if provided
if [ -n "$CLAUDE_SESSION_ID" ]; then
    CLAUDE_CMD="$CLAUDE_CMD --resume \"$CLAUDE_SESSION_ID\""
fi

# Log file for this run
RUN_LOG="$PROJECT_DIR/.claude/context/last-run.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Starting headless Claude run at $TIMESTAMP"
echo "Project: $PROJECT_DIR"
echo "Prompt: ${PROMPT:0:100}..."
echo ""

# Run claude and capture output
eval $CLAUDE_CMD 2>&1 | tee "$RUN_LOG"

echo ""
echo "Run complete. Output saved to $RUN_LOG"
