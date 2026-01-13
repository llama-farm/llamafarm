#!/bin/bash
#
# Slack Notification Hook
#
# Mock mode: Logs to file when CLAUDE_SLACK_WEBHOOK is not set
# Real mode: Sends to Slack when webhook URL is provided
#
# Usage: Receives JSON notification payload via stdin
#

set -e

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CONTEXT_DIR="$PROJECT_DIR/.claude/context"
LOG_FILE="$CONTEXT_DIR/notifications.log"

# Ensure context directory exists
mkdir -p "$CONTEXT_DIR"

# Read input from stdin
INPUT=$(cat)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Extract notification type if available
NOTIFICATION_TYPE=$(echo "$INPUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('type', 'general'))" 2>/dev/null || echo "general")

if [ -z "$CLAUDE_SLACK_WEBHOOK" ]; then
    # Mock mode - log to file
    echo "[$TIMESTAMP] [$NOTIFICATION_TYPE] $INPUT" >> "$LOG_FILE"
    echo "[MOCK] Notification logged to $LOG_FILE"
else
    # Real mode - send to Slack
    # Format the payload for Slack if it's not already formatted
    if echo "$INPUT" | python3 -c "import sys, json; d=json.load(sys.stdin); exit(0 if 'text' in d or 'blocks' in d else 1)" 2>/dev/null; then
        # Already formatted for Slack
        PAYLOAD="$INPUT"
    else
        # Wrap in basic Slack message format
        PAYLOAD=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    msg = data.get('message', str(data))
except:
    msg = sys.stdin.read()
print(json.dumps({'text': msg}))
")
    fi

    # Send to Slack
    RESPONSE=$(curl -s -X POST -H 'Content-type: application/json' \
        --data "$PAYLOAD" \
        "$CLAUDE_SLACK_WEBHOOK" 2>&1)

    if [ "$RESPONSE" = "ok" ]; then
        echo "[SLACK] Notification sent successfully"
    else
        echo "[SLACK] Response: $RESPONSE"
    fi

    # Also log to file for record keeping
    echo "[$TIMESTAMP] [SENT] [$NOTIFICATION_TYPE] $INPUT" >> "$LOG_FILE"
fi

exit 0
