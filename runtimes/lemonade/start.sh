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
LEMONADE_CONTEXT_SIZE="${LEMONADE_CONTEXT_SIZE:-65536}"  # Default context size (doubled)
LEMONADE_MODEL="${LEMONADE_MODEL:-}"  # Optional: pre-load a specific model

# Try to read config from project llamafarm.yaml if available
# This allows the startup script to respect project configuration
# Try multiple possible locations for llamafarm.yaml
# Parse --config-file/-f and --model-name/-m arguments in any order
LF_CONFIG_FILE=""
lf_model_name=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-file|-f)
            LF_CONFIG_FILE="$2"
            shift 2
            ;;
        --model-name|-m)
            lf_model_name="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# If --model-name was provided, propagate it to LF_MODEL_NAME (CLI arg should take precedence over env)
if [ -n "$lf_model_name" ]; then
    LF_MODEL_NAME="$lf_model_name"
fi

# Search for llamafarm.yaml in multiple locations
if [ -z "$LF_CONFIG_FILE" ]; then
    # Try current directory first
    if [ -f "./llamafarm.yaml" ]; then
        LF_CONFIG_FILE="./llamafarm.yaml"
    # Try parent directory (in case we're in config/ or cli/)
    elif [ -f "../llamafarm.yaml" ]; then
        LF_CONFIG_FILE="../llamafarm.yaml"
    # Try two levels up (in case we're deeper)
    elif [ -f "../../llamafarm.yaml" ]; then
        LF_CONFIG_FILE="../../llamafarm.yaml"
    fi
fi

if [ -z "$LF_CONFIG_FILE" ] || [ ! -f "$LF_CONFIG_FILE" ]; then
    echo "Error: llamafarm.yaml not found. Searched:"
    echo "  - ./llamafarm.yaml"
    echo "  - ../llamafarm.yaml"
    echo "  - ../../llamafarm.yaml"
    echo ""
    echo "Use --config-file <path> to specify location"
    exit 1
fi

if [ -n "$LF_CONFIG_FILE" ]; then
    # Use uv run python to parse YAML (uses project's venv)
    # Parse multi-model format
    # If LF_MODEL_NAME env var is set, find that specific model
    # Otherwise, find first lemonade model in list
    if [ -n "$LF_MODEL_NAME" ]; then
        CONFIG_MODEL=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('model', ''))
        break
" 2>/dev/null)

        CONFIG_PORT=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('provider_config', {}).get('port', ''))
        break
" 2>/dev/null)

        CONFIG_BACKEND=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('provider_config', {}).get('backend', ''))
        break
" 2>/dev/null)

        CONFIG_CONTEXT_SIZE=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('provider_config', {}).get('context_size', ''))
        break
" 2>/dev/null)

        CONFIG_CHECKPOINT=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('provider_config', {}).get('checkpoint', ''))
        break
" 2>/dev/null)

        CONFIG_RECIPE=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade' and model_config.get('name') == '$LF_MODEL_NAME':
        print(model_config.get('provider_config', {}).get('recipe', ''))
        break
" 2>/dev/null)

    else
        # No lf_model_name specified, use first lemonade model
        CONFIG_MODEL=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('model', ''))
        break
" 2>/dev/null)

        CONFIG_PORT=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('provider_config', {}).get('port', ''))
        break
" 2>/dev/null)

        CONFIG_BACKEND=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('provider_config', {}).get('backend', ''))
        break
" 2>/dev/null)

        CONFIG_CONTEXT_SIZE=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('provider_config', {}).get('context_size', ''))
        break
" 2>/dev/null)

        CONFIG_CHECKPOINT=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('provider_config', {}).get('checkpoint', ''))
        break
" 2>/dev/null)

        CONFIG_RECIPE=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$LF_CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
for model_config in models:
    if model_config.get('provider') == 'lemonade':
        print(model_config.get('provider_config', {}).get('recipe', ''))
        break
" 2>/dev/null)

    fi


    # Override defaults with config values if present
    [ -n "$CONFIG_PORT" ] && LEMONADE_PORT="$CONFIG_PORT"
    [ -n "$CONFIG_BACKEND" ] && LEMONADE_BACKEND="$CONFIG_BACKEND"
    [ -n "$CONFIG_CONTEXT_SIZE" ] && LEMONADE_CONTEXT_SIZE="$CONFIG_CONTEXT_SIZE"
    [ -n "$CONFIG_MODEL" ] && [ -z "$LEMONADE_MODEL" ] && LEMONADE_MODEL="$CONFIG_MODEL"
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

# Check if lemonade-server-dev is available, install if not
echo -e "${GREEN}Checking Lemonade SDK...${NC}"

# Check if lemonade-sdk is installed
if ! uv pip show lemonade-sdk &>/dev/null; then
    echo -e "${YELLOW}Lemonade SDK not found. Installing...${NC}"
    uv pip install lemonade-sdk || {
        echo -e "${RED}Failed to install Lemonade SDK${NC}"
        echo "Try running: uv pip install lemonade-sdk"
        echo "For more info: https://lemonade-server.ai/"
        exit 1
    }
    echo -e "${GREEN}Lemonade SDK installed successfully!${NC}"
fi

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
    echo "Example: LEMONADE_MODEL=user.Qwen3-4B nx start lemonade"
    echo ""
    echo "To download models, use:"
    echo "  uv run lemonade-server-dev pull user.Qwen3-4B --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M --recipe llamacpp"
    echo ""
    exit 1
fi

# Build lemonade-server-dev command
# Using serve subcommand (NOT run - serve is the correct command)
# Note: PyPI package provides lemonade-server-dev, installer provides lemonade-server
LEMONADE_CMD="uv run lemonade-server-dev serve --port $LEMONADE_PORT --host $LEMONADE_HOST --ctx-size $LEMONADE_CONTEXT_SIZE --no-tray"

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
