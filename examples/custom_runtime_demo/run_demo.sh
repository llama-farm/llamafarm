#!/bin/bash
set -e

# Get absolute path to this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_PATH="$DIR/llamafarm.yaml"

echo "Using config: $CONFIG_PATH"

# Check if LF CLI is available
if ! command -v lf &> /dev/null; then
    echo "Error: 'lf' command not found. Please build the CLI first."
    exit 1
fi

echo "Starting chat session..."
echo "Try: 'Reverse the string Hello World'"
echo "Try: 'Calculate factorial of 5'"

# Run lf chat with the custom config
export LF_CONFIG_PATH="$CONFIG_PATH"

echo "--- Query 1: String Reversal ---"
lf chat --cwd "$DIR" --auto-start=false "Reverse the string 'Hello LlamaFarm' and tell me the length"

echo ""
echo "--- Query 2: Factorial Calculation ---"
lf chat --cwd "$DIR" --auto-start=false "Calculate factorial of 5"
