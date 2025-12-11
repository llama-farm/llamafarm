#!/usr/bin/env bash
# Generate a detailed prose changelog from conventional commits using LlamaFarm
#
# This script is used by the prose-changelog GitHub Action but can also be run
# manually for local development.
#
# Environment variables (set by action or manually):
#   INPUT_VERSION         - Version to generate changelog for (empty = latest)
#   INPUT_MODEL           - LlamaFarm model to use
#   INPUT_CHANGELOG_FILE  - Path to CHANGELOG.md (default: CHANGELOG.md)
#   OPENAI_API_KEY        - Passed through to LlamaFarm for API auth
#
# Manual usage:
#   ./generate.sh                                    # Latest version, defaults
#   INPUT_VERSION=0.0.19 ./generate.sh               # Specific version
#   INPUT_MODEL=powerful ./generate.sh               # Use specific model

set -euo pipefail

# Configuration from environment or defaults
VERSION="${INPUT_VERSION:-}"
MODEL="${INPUT_MODEL:-}"
CHANGELOG_FILE="${INPUT_CHANGELOG_FILE:-CHANGELOG.md}"

# Resolve changelog file path
if [[ ! -f "$CHANGELOG_FILE" ]]; then
    # Try relative to repo root
    if [[ -n "$GITHUB_WORKSPACE" ]] && [[ -f "$GITHUB_WORKSPACE/$CHANGELOG_FILE" ]]; then
        CHANGELOG_FILE="$GITHUB_WORKSPACE/$CHANGELOG_FILE"
    else
        echo "Error: CHANGELOG.md not found at $CHANGELOG_FILE" >&2
        exit 1
    fi
fi

# If no version specified, extract the latest from CHANGELOG.md
if [[ -z "$VERSION" ]]; then
    VERSION=$(grep -m1 '## \[' "$CHANGELOG_FILE" | sed 's/.*\[\([^]]*\)\].*/\1/')
    if [[ -z "$VERSION" ]]; then
        echo "Error: Could not determine latest version from CHANGELOG.md" >&2
        exit 1
    fi
    echo "Using latest version: $VERSION" >&2
fi

# Extract the changelog section for the specified version
CHANGELOG_SECTION=$(awk -v ver="$VERSION" '
    /^## \[/ {
        if (found) exit
        if (index($0, "[" ver "]")) found=1
    }
    found { print }
' "$CHANGELOG_FILE")

if [[ -z "$CHANGELOG_SECTION" ]]; then
    echo "Error: Could not find changelog section for version $VERSION" >&2
    exit 1
fi

echo "Extracted changelog for version $VERSION" >&2

# Create temporary directory for LlamaFarm project
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Get the directory where this script lives (contains llamafarm.yaml)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy the LlamaFarm project configuration
cp "$SCRIPT_DIR/llamafarm.yaml" "$TEMP_DIR/llamafarm.yaml"

# Build the prompt
read -r -d '' PROMPT << EOM || true
Please transform the following conventional commits changelog into detailed,
user-friendly prose release notes. Focus on explaining the value and impact of each
change for end users.

Input changelog:

${CHANGELOG_SECTION}
EOM

# Check if lf CLI is available, install if not (for CI)
if ! command -v lf &> /dev/null; then
    echo "Installing LlamaFarm CLI..." >&2
    curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
    export PATH="$HOME/.llamafarm/bin:$PATH"
fi

# Build lf command
LF_CMD="lf chat --no-rag"
if [[ -n "$MODEL" ]]; then
    LF_CMD="$LF_CMD --model $MODEL"
fi

# Change to temp directory and generate
cd "$TEMP_DIR"

echo "Generating prose changelog..." >&2
echo "Using config: $TEMP_DIR/llamafarm.yaml" >&2

# Write prompt to a temp file to avoid shell escaping issues with large text
PROMPT_FILE="$TEMP_DIR/prompt.txt"
echo "$PROMPT" > "$PROMPT_FILE"

# Run lf chat with input from file, capture both stdout and stderr
set +e  # Don't exit on error so we can capture the output
PROSE_CHANGELOG=$($LF_CMD -f "$PROMPT_FILE" 2>&1)
LF_EXIT_CODE=$?
set -e

if [[ $LF_EXIT_CODE -ne 0 ]] || [[ -z "$PROSE_CHANGELOG" ]]; then
    echo "Error: Failed to generate prose changelog (exit code: $LF_EXIT_CODE)" >&2
    echo "Command output:" >&2
    echo "$PROSE_CHANGELOG" >&2
    echo "" >&2
    echo "Troubleshooting:" >&2
    echo "  - For universal provider: ensure 'lf start' is running" >&2
    echo "  - For OpenAI: ensure OPENAI_API_KEY is set" >&2
    echo "  - Check llamafarm.yaml configuration" >&2
    exit 1
fi

# Save to file
OUTPUT_FILE="$TEMP_DIR/prose-changelog-${VERSION}.md"
echo "$PROSE_CHANGELOG" > "$OUTPUT_FILE"

# Set outputs for GitHub Actions
if [[ -n "$GITHUB_OUTPUT" ]]; then
    echo "version=$VERSION" >> "$GITHUB_OUTPUT"
    echo "output_file=$OUTPUT_FILE" >> "$GITHUB_OUTPUT"

    # Multi-line output
    {
        echo "prose_changelog<<EOF"
        echo "$PROSE_CHANGELOG"
        echo "EOF"
    } >> "$GITHUB_OUTPUT"
fi

# Copy to workspace for artifact upload
if [[ -n "$GITHUB_WORKSPACE" ]]; then
    cp "$OUTPUT_FILE" "$GITHUB_WORKSPACE/prose-changelog-${VERSION}.md"
    echo "output_file=$GITHUB_WORKSPACE/prose-changelog-${VERSION}.md" >> "$GITHUB_OUTPUT"
fi

# Output to stdout
echo ""
echo "$PROSE_CHANGELOG"
