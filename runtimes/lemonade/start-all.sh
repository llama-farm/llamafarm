#!/bin/bash
set -e

# Start All Lemonade Models Script
# This script reads llamafarm.yaml and starts a Lemonade server for each lemonade model

# Find llamafarm.yaml
CONFIG_FILE=""
if [ -f "../../llamafarm.yaml" ]; then
    CONFIG_FILE="../../llamafarm.yaml"
elif [ -f "../llamafarm.yaml" ]; then
    CONFIG_FILE="../llamafarm.yaml"
elif [ -f "llamafarm.yaml" ]; then
    CONFIG_FILE="llamafarm.yaml"
fi

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: llamafarm.yaml not found"
    exit 1
fi

# Get all Lemonade model details from config
MODEL_INFO=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
lemonade_models = [m for m in models if m.get('provider') == 'lemonade']
for model in lemonade_models:
    name = model.get('name', '')
    model_id = model.get('model', '')
    port = model.get('lemonade', {}).get('port', 'default')
    auto_download = model.get('lemonade', {}).get('auto_download', False)
    checkpoint = model.get('lemonade', {}).get('checkpoint', '')
    print(f'{name}|{model_id}|{port}|{auto_download}|{checkpoint}')
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

while IFS='|' read -r MODEL_NAME MODEL_ID PORT AUTO_DOWNLOAD CHECKPOINT; do
    MODEL_COUNT=$((MODEL_COUNT + 1))

    echo -e "${CYAN}[$MODEL_COUNT] $MODEL_NAME${NC}"
    echo "    Model:  $MODEL_ID"
    echo "    Port:   $PORT"

    if [ "$AUTO_DOWNLOAD" = "True" ]; then
        echo -e "    ${YELLOW}Auto-download: enabled${NC}"
        if [ -n "$CHECKPOINT" ]; then
            echo "    Checkpoint: $CHECKPOINT"
        fi
    fi

    echo "    Logs:   /tmp/lemonade-$MODEL_NAME.log"

    LEMONADE_MODEL_NAME=$MODEL_NAME bash ../runtimes/lemonade/start.sh > /tmp/lemonade-$MODEL_NAME.log 2>&1 &
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
