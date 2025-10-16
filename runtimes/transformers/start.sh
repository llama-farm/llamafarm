#!/bin/bash
set -e

# Transformers Runtime Startup Script
# Auto-installs dependencies and starts the FastAPI server

# Configuration with sensible defaults
TRANSFORMERS_PORT="${TRANSFORMERS_PORT:-11540}"
TRANSFORMERS_HOST="${TRANSFORMERS_HOST:-127.0.0.1}"
TRANSFORMERS_OUTPUT_DIR="${TRANSFORMERS_OUTPUT_DIR:-$HOME/.llamafarm/outputs/images}"
TRANSFORMERS_CACHE_DIR="${TRANSFORMERS_CACHE_DIR:-$HOME/.cache/huggingface}"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Transformers Runtime Server ===${NC}"
echo "Port: $TRANSFORMERS_PORT"
echo "Host: $TRANSFORMERS_HOST"
echo "Output Directory: $TRANSFORMERS_OUTPUT_DIR"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    DEVICE="mps"
elif command -v nvidia-smi &> /dev/null; then
    PLATFORM="Linux (NVIDIA)"
    DEVICE="cuda"
else
    PLATFORM="Linux/Other"
    DEVICE="cpu"
fi

echo "Platform: $PLATFORM"
echo "Expected Device: $DEVICE"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found${NC}"
    exit 1
fi

# Check/install dependencies
echo -e "${GREEN}Checking dependencies...${NC}"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Please install it first:${NC}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Use uv to sync dependencies from pyproject.toml
echo -e "${YELLOW}Syncing dependencies with uv...${NC}"
uv sync

# Platform-specific dependencies
if [[ "$DEVICE" == "cuda" ]]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Install xformers on non-macOS platforms (optional optimization)
if [[ "$PLATFORM" != "macOS" ]]; then
    echo -e "${YELLOW}Installing xformers (optional optimization)...${NC}"
    uv pip install xformers || echo -e "${YELLOW}xformers installation failed (optional, continuing...)${NC}"
fi

# Create output directory
mkdir -p "$TRANSFORMERS_OUTPUT_DIR"

echo -e "${GREEN}All dependencies ready!${NC}"
echo ""

# Start server
echo -e "${GREEN}Starting Transformers Runtime...${NC}"
echo ""
echo "OpenAI-compatible endpoints:"
echo "  POST http://$TRANSFORMERS_HOST:$TRANSFORMERS_PORT/v1/chat/completions"
echo "  POST http://$TRANSFORMERS_HOST:$TRANSFORMERS_PORT/v1/images/generations"
echo "  POST http://$TRANSFORMERS_HOST:$TRANSFORMERS_PORT/v1/images/edits"
echo "  GET  http://$TRANSFORMERS_HOST:$TRANSFORMERS_PORT/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Export environment variables
export TRANSFORMERS_PORT
export TRANSFORMERS_HOST
export TRANSFORMERS_OUTPUT_DIR
export HF_HOME="$TRANSFORMERS_CACHE_DIR"

# Pass through device override variables if set
if [[ -n "$TRANSFORMERS_SKIP_MPS" ]]; then
    export TRANSFORMERS_SKIP_MPS
fi
if [[ -n "$TRANSFORMERS_FORCE_CPU" ]]; then
    export TRANSFORMERS_FORCE_CPU
fi

# Start server with uv
exec uv run python server.py
