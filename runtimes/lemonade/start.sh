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
LEMONADE_CONTEXT_SIZE="${LEMONADE_CONTEXT_SIZE:-32768}"  # Default context size
LEMONADE_MODEL="${LEMONADE_MODEL:-}"  # Optional: pre-load a specific model

# Try to read config from project llamafarm.yaml if available
# This allows the startup script to respect project configuration
if [ -f "../llamafarm.yaml" ]; then
    # Use uv run python to parse YAML (uses project's venv)
    CONFIG_PORT=$(uv run python -c "import yaml; config = yaml.safe_load(open('../llamafarm.yaml')); print(config.get('runtime', {}).get('lemonade', {}).get('port', ''))" 2>/dev/null)
    CONFIG_BACKEND=$(uv run python -c "import yaml; config = yaml.safe_load(open('../llamafarm.yaml')); print(config.get('runtime', {}).get('lemonade', {}).get('backend', ''))" 2>/dev/null)
    CONFIG_CONTEXT_SIZE=$(uv run python -c "import yaml; config = yaml.safe_load(open('../llamafarm.yaml')); print(config.get('runtime', {}).get('lemonade', {}).get('context_size', ''))" 2>/dev/null)
    CONFIG_MODEL=$(uv run python -c "import yaml; config = yaml.safe_load(open('../llamafarm.yaml')); print(config.get('runtime', {}).get('model', ''))" 2>/dev/null)
    CONFIG_HF_TOKEN=$(uv run python -c "import yaml; config = yaml.safe_load(open('../llamafarm.yaml')); print(config.get('runtime', {}).get('huggingface_token', ''))" 2>/dev/null)

    # Override defaults with config values if present
    [ -n "$CONFIG_PORT" ] && LEMONADE_PORT="$CONFIG_PORT"
    [ -n "$CONFIG_BACKEND" ] && LEMONADE_BACKEND="$CONFIG_BACKEND"
    [ -n "$CONFIG_CONTEXT_SIZE" ] && LEMONADE_CONTEXT_SIZE="$CONFIG_CONTEXT_SIZE"
    [ -n "$CONFIG_MODEL" ] && [ -z "$LEMONADE_MODEL" ] && LEMONADE_MODEL="$CONFIG_MODEL"

    # Export Hugging Face token if provided (needed for downloading gated models)
    [ -n "$CONFIG_HF_TOKEN" ] && export HF_TOKEN="$CONFIG_HF_TOKEN"
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

# Check if lemonade-server-dev is available
# Note: We don't do a full check here, let uv handle it
echo -e "${GREEN}Checking Lemonade SDK...${NC}"

# Check if port is already in use
if lsof -Pi :$LEMONADE_PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}WARNING: Port $LEMONADE_PORT is already in use${NC}"
    echo "Another Lemonade instance may be running, or the port is occupied."
    echo ""
    echo "Attempting to start anyway (will fail if port is truly unavailable)..."
fi

# Model is required for Lemonade
if [ -z "$LEMONADE_MODEL" ]; then
    echo -e "${RED}ERROR: Model not specified${NC}"
    echo ""
    echo "Set LEMONADE_MODEL environment variable or configure runtime.model in llamafarm.yaml"
    echo ""
    echo "Example: LEMONADE_MODEL=Qwen3-0.6B-GGUF nx start lemonade"
    echo ""
    exit 1
fi

# Build lemonade-server-dev command
# Using run subcommand which loads the model
LEMONADE_CMD="uv run lemonade-server-dev run $LEMONADE_MODEL --port $LEMONADE_PORT --host $LEMONADE_HOST --no-tray"

# Configure backend-specific options
case "$LEMONADE_BACKEND" in
    llamacpp)
        # llamacpp: Detect platform and available accelerators
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS: Use Metal for Apple Silicon/Intel GPUs
            LEMONADE_CMD="$LEMONADE_CMD --llamacpp metal --ctx-size $LEMONADE_CONTEXT_SIZE"
        elif command -v nvidia-smi &> /dev/null; then
            # Linux with NVIDIA GPU detected: Use CUDA (better than Vulkan for NVIDIA)
            # Note: Requires llamacpp built with CUDA support
            echo "Detected NVIDIA GPU, attempting to use CUDA acceleration..."
            LEMONADE_CMD="$LEMONADE_CMD --llamacpp cuda --ctx-size $LEMONADE_CONTEXT_SIZE"
        elif [[ -d /dev/dri ]]; then
            # Linux with GPU (AMD/Intel): Use Vulkan
            echo "Detected GPU device, using Vulkan acceleration..."
            LEMONADE_CMD="$LEMONADE_CMD --llamacpp vulkan --ctx-size $LEMONADE_CONTEXT_SIZE"
        else
            # Fallback to CPU if no GPU detected
            echo "No GPU detected, using CPU-only mode..."
            LEMONADE_CMD="$LEMONADE_CMD --llamacpp cpu --ctx-size $LEMONADE_CONTEXT_SIZE"
        fi
        ;;

    transformers)
        # transformers: PyTorch/HuggingFace backend (uses PyTorch's CUDA/MPS auto-detection)
        echo "Using transformers backend (PyTorch will auto-detect GPU acceleration)..."
        LEMONADE_CMD="$LEMONADE_CMD --transformers"
        ;;

    onnx|*)
        # onnx: Default backend, works on all systems
        # ONNX automatically uses available execution providers (CUDA, DirectML, CoreML, etc.)
        echo "Using ONNX backend (default, will auto-detect acceleration)..."
        # No additional flags needed - ONNX runtime auto-detects available providers
        ;;
esac

# Prevent browser from opening
export BROWSER=true

echo ""
echo -e "${GREEN}Starting Lemonade Server with model: $LEMONADE_MODEL${NC}"
echo "Command: $LEMONADE_CMD"
echo ""
echo "Once started, Lemonade will be available at:"
echo "  http://$LEMONADE_HOST:$LEMONADE_PORT/api/v1"
echo ""
echo "OpenAI-compatible endpoints:"
echo "  POST http://$LEMONADE_HOST:$LEMONADE_PORT/api/v1/chat/completions"
echo "  POST http://$LEMONADE_HOST:$LEMONADE_PORT/api/v1/completions"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Lemonade Server
exec $LEMONADE_CMD
