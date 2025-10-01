#!/bin/bash
set -e

# Lemonade Server Startup Script
# This script starts the Lemonade inference server with OpenAI-compatible API
# Designed to be extensible for future runtime additions (vLLM, etc.)

# Configuration with sensible defaults
# These can be overridden by environment variables or project config
LEMONADE_PORT="${LEMONADE_PORT:-11534}"
LEMONADE_HOST="${LEMONADE_HOST:-127.0.0.1}"
LEMONADE_BACKEND="${LEMONADE_BACKEND:-onnx}"  # Default to ONNX (works on all systems)
LEMONADE_MODEL="${LEMONADE_MODEL:-}"  # Optional: pre-load a specific model

# Try to read config from project seed if available
# This allows the startup script to respect project configuration
if [ -f "server/seeds/project_seed/llamafarm.yaml" ]; then
    # Check if yq is available for YAML parsing
    if command -v yq &> /dev/null; then
        CONFIG_PORT=$(yq eval '.runtime.lemonade.port // ""' server/seeds/project_seed/llamafarm.yaml 2>/dev/null)
        CONFIG_BACKEND=$(yq eval '.runtime.lemonade.backend // ""' server/seeds/project_seed/llamafarm.yaml 2>/dev/null)
        CONFIG_MODEL=$(yq eval '.runtime.model // ""' server/seeds/project_seed/llamafarm.yaml 2>/dev/null)

        # Override defaults with config values if present
        [ -n "$CONFIG_PORT" ] && LEMONADE_PORT="$CONFIG_PORT"
        [ -n "$CONFIG_BACKEND" ] && LEMONADE_BACKEND="$CONFIG_BACKEND"
        [ -n "$CONFIG_MODEL" ] && [ -z "$LEMONADE_MODEL" ] && LEMONADE_MODEL="$CONFIG_MODEL"
    fi
fi

# Color output for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Lemonade Runtime Server ===${NC}"
echo "Port: $LEMONADE_PORT"
echo "Host: $LEMONADE_HOST"
echo "Backend: $LEMONADE_BACKEND"

# Check if lemonade is installed
if ! command -v lemonade &> /dev/null; then
    echo -e "${RED}ERROR: Lemonade not found${NC}"
    echo ""
    echo "Please install Lemonade SDK:"
    echo "  pip install lemonade-sdk"
    echo ""
    echo "Or install via uv (recommended for LlamaFarm):"
    echo "  uv pip install lemonade-sdk"
    echo ""
    echo "For more information: https://lemonade-server.ai/docs/"
    exit 1
fi

echo -e "${GREEN}Lemonade SDK found${NC}"

# Check if port is already in use
if lsof -Pi :$LEMONADE_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}WARNING: Port $LEMONADE_PORT is already in use${NC}"
    echo "Another Lemonade instance may be running, or the port is occupied."
    echo "You can change the port by setting LEMONADE_PORT environment variable."
    echo ""
    echo "Attempting to start anyway (Lemonade will error if port is truly unavailable)..."
fi

# Build lemonade command
LEMONADE_CMD="lemonade server start --port $LEMONADE_PORT --host $LEMONADE_HOST"

# Add backend selection
if [ -n "$LEMONADE_BACKEND" ]; then
    LEMONADE_CMD="$LEMONADE_CMD --backend $LEMONADE_BACKEND"
fi

# Add model if specified
if [ -n "$LEMONADE_MODEL" ]; then
    echo "Pre-loading model: $LEMONADE_MODEL"
    LEMONADE_CMD="$LEMONADE_CMD --model $LEMONADE_MODEL"
fi

echo ""
echo -e "${GREEN}Starting Lemonade Server...${NC}"
echo "Command: $LEMONADE_CMD"
echo ""
echo "Once started, Lemonade will be available at:"
echo "  http://$LEMONADE_HOST:$LEMONADE_PORT/v1"
echo ""
echo "OpenAI-compatible endpoints:"
echo "  POST http://$LEMONADE_HOST:$LEMONADE_PORT/v1/chat/completions"
echo "  POST http://$LEMONADE_HOST:$LEMONADE_PORT/v1/completions"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Lemonade Server
exec $LEMONADE_CMD
