#!/usr/bin/env bash

# LlamaFarm Jetson Start Script
# Starts the Universal Runtime with Jetson-optimized settings
#
# Usage:
#   ./deployment/jetson/start.sh [model] [ctx-len]
#
# Examples:
#   ./deployment/jetson/start.sh                                    # Use defaults
#   ./deployment/jetson/start.sh 'unsloth/Qwen3-0.6B-GGUF:Q4_K_M'  # Custom model
#   ./deployment/jetson/start.sh 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' 4096  # Custom ctx

set -e

# Defaults
MODEL="${1:-unsloth/Qwen3-1.7B-GGUF:Q4_K_M}"
CTX_LEN="${2:-2048}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/runtimes/universal"

# Source Jetson environment if exists
if [[ -f "$RUNTIME_DIR/.env.jetson" ]]; then
    source "$RUNTIME_DIR/.env.jetson"
fi

# Verify venv exists
if [[ ! -f "$RUNTIME_DIR/.venv/bin/python" ]]; then
    echo "Error: Virtual environment not found. Run setup.sh first:"
    echo "  ./deployment/jetson/setup.sh"
    exit 1
fi

cd "$RUNTIME_DIR"

echo "Starting LlamaFarm Universal Runtime (Jetson)"
echo "  Model: $MODEL"
echo "  Context: $CTX_LEN"
echo ""

exec .venv/bin/python server.py --model "$MODEL" --ctx-len "$CTX_LEN"
