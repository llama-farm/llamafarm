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

# Get all Lemonade model names from config
MODEL_NAMES=$(uv run python -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
models = config.get('runtime', {}).get('models', [])
lemonade_models = [m.get('name') for m in models if m.get('provider') == 'lemonade']
for name in lemonade_models:
    print(name)
" 2>/dev/null)

if [ -z "$MODEL_NAMES" ]; then
    echo "No Lemonade models found in $CONFIG_FILE"
    exit 1
fi

# Start each Lemonade model in the background
PIDS=()
MODEL_COUNT=0

for MODEL_NAME in $MODEL_NAMES; do
    echo "Starting Lemonade model: $MODEL_NAME"
    LEMONADE_MODEL_NAME=$MODEL_NAME bash ../runtimes/lemonade/start.sh > /tmp/lemonade-$MODEL_NAME.log 2>&1 &
    PIDS+=($!)
    MODEL_COUNT=$((MODEL_COUNT + 1))
    echo "  PID: $! (logs: /tmp/lemonade-$MODEL_NAME.log)"
    sleep 2  # Stagger startup
done

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
