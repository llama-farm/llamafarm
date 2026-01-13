#!/bin/bash
#
# Post-Edit Lint Hook
#
# Runs linters after file edits. Currently supports:
#   - Python files: ruff check --fix && ruff format
#   - JavaScript/TypeScript: prettier (if available)
#
# Receives tool input via stdin with file_path information
#

set -e

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Read input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Handle different input structures
    if 'tool_input' in data:
        print(data['tool_input'].get('file_path', ''))
    elif 'file_path' in data:
        print(data['file_path'])
    else:
        print('')
except Exception as e:
    print('', file=sys.stderr)
" 2>/dev/null)

# Exit if no file path
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    exit 0
fi

# Get file extension
EXT="${FILE_PATH##*.}"

case "$EXT" in
    py)
        # Python files - use ruff if available
        if command -v ruff &> /dev/null; then
            echo "[LINT] Running ruff on $FILE_PATH"
            ruff check --fix "$FILE_PATH" 2>/dev/null || true
            ruff format "$FILE_PATH" 2>/dev/null || true
        elif command -v black &> /dev/null; then
            echo "[LINT] Running black on $FILE_PATH"
            black --quiet "$FILE_PATH" 2>/dev/null || true
        fi
        ;;
    js|jsx|ts|tsx|json|md|yaml|yml)
        # JavaScript/TypeScript/JSON/Markdown - use prettier if available
        if command -v prettier &> /dev/null; then
            echo "[LINT] Running prettier on $FILE_PATH"
            prettier --write "$FILE_PATH" 2>/dev/null || true
        fi
        ;;
    go)
        # Go files - use gofmt if available
        if command -v gofmt &> /dev/null; then
            echo "[LINT] Running gofmt on $FILE_PATH"
            gofmt -w "$FILE_PATH" 2>/dev/null || true
        fi
        ;;
    rs)
        # Rust files - use rustfmt if available
        if command -v rustfmt &> /dev/null; then
            echo "[LINT] Running rustfmt on $FILE_PATH"
            rustfmt "$FILE_PATH" 2>/dev/null || true
        fi
        ;;
    *)
        # Unknown extension - skip
        ;;
esac

exit 0
