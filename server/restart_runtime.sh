#!/bin/bash
# Script to restart the universal-runtime service
# This runs in the background so it doesn't block the server

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtimes/universal"
RUNTIME_PORT=11540
LOG_FILE="/tmp/runtime-restart.log"

echo "[$(date)] Starting universal-runtime restart..." > "$LOG_FILE"

# Kill existing process on port 11540
PID=$(lsof -ti:$RUNTIME_PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "[$(date)] Killing existing process (PID: $PID)" >> "$LOG_FILE"
    kill -9 $PID 2>/dev/null
    sleep 2
fi

# Start the runtime in the background
echo "[$(date)] Starting universal-runtime..." >> "$LOG_FILE"
cd "$RUNTIME_DIR"
nohup uv run python server.py >> "$LOG_FILE" 2>&1 &
NEW_PID=$!

echo "[$(date)] Started universal-runtime (PID: $NEW_PID)" >> "$LOG_FILE"

# Wait a moment and check if it's healthy
sleep 5
HEALTH=$(curl -s http://localhost:$RUNTIME_PORT/health 2>/dev/null | grep -o '"status":"healthy"')
if [ -n "$HEALTH" ]; then
    echo "[$(date)] Universal-runtime is healthy!" >> "$LOG_FILE"
else
    echo "[$(date)] WARNING: Universal-runtime may not be healthy yet" >> "$LOG_FILE"
fi

exit 0
