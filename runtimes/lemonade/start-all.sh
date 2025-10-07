#!/bin/bash
set -e

# Start All Lemonade Models Script
# This script reads llamafarm.yaml and starts a Lemonade server for each lemonade model

# Resolve script directory to reliably invoke sibling start.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments similar to start.sh to find config file
LF_CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-file|-f)
            LF_CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

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

CONFIG_FILE="$LF_CONFIG_FILE"

# Get all Lemonade model details from config
MODEL_INFO=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
lemonade_models = [m for m in models if m.get('provider') == 'lemonade']
for model in lemonade_models:
    name = model.get('name', '')
    model_id = model.get('model', '')
    port = model.get('provider_config', {}).get('port', 'default')
    print(f'{name}|{model_id}|{port}')
" 2>/dev/null)

if [ -z "$MODEL_INFO" ]; then
    echo "No Lemonade models found in $CONFIG_FILE"
    exit 1
fi

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${GREEN}=== Starting Lemonade Models ===${NC}"
echo ""

# Start each Lemonade model in the background
PIDS=()
MODEL_COUNT=0

while IFS='|' read -r MODEL_NAME MODEL_ID PORT; do
    MODEL_COUNT=$((MODEL_COUNT + 1))

    echo -e "${CYAN}[$MODEL_COUNT] $MODEL_NAME${NC}"
    echo "    Model:  $MODEL_ID"
    echo "    Port:   $PORT"
    echo "    Logs:   /tmp/lemonade-$MODEL_NAME.log"

    LF_MODEL_NAME="$MODEL_NAME" CONFIG_FILE="$CONFIG_FILE" bash "$SCRIPT_DIR/start.sh" --config-file "$CONFIG_FILE" > /tmp/lemonade-$MODEL_NAME.log 2>&1 &
    PIDS+=($!)
    echo "    PID:    $!"
    echo ""

    sleep 2  # Stagger startup
done <<< "$MODEL_INFO"

echo ""
echo "Started $MODEL_COUNT Lemonade model(s)"
echo "PIDs: ${PIDS[@]}"
echo ""

# If running interactively (not from nx), wait for processes
# Otherwise, let them run in background
if [ -t 0 ]; then
    echo "Press Ctrl+C to stop all"
    # Wait for all background processes
    wait
else
    echo "Running in background - check logs in /tmp/lemonade-*.log"
fi
