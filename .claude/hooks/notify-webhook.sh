#!/bin/bash
#
# Generic Webhook Notification Hook
#
# Sends notifications to any webhook endpoint.
# Configure via environment variables:
#   CLAUDE_WEBHOOK_URL - The webhook endpoint
#   CLAUDE_WEBHOOK_AUTH - Optional auth header value (e.g., "Bearer token123")
#   CLAUDE_WEBHOOK_METHOD - HTTP method (default: POST)
#
# Usage: Receives JSON notification payload via stdin
#

set -e

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CONTEXT_DIR="$PROJECT_DIR/.claude/context"
LOG_FILE="$CONTEXT_DIR/webhook-notifications.log"

# Ensure context directory exists
mkdir -p "$CONTEXT_DIR"

# Read input from stdin
INPUT=$(cat)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Default method
METHOD="${CLAUDE_WEBHOOK_METHOD:-POST}"

if [ -z "$CLAUDE_WEBHOOK_URL" ]; then
    # No webhook configured - just log
    echo "[$TIMESTAMP] [NO_WEBHOOK] $INPUT" >> "$LOG_FILE"
    echo "[WEBHOOK] No CLAUDE_WEBHOOK_URL configured, notification logged only"
    exit 0
fi

# Build curl command
CURL_ARGS=(-s -X "$METHOD" -H 'Content-Type: application/json')

# Add auth header if provided
if [ -n "$CLAUDE_WEBHOOK_AUTH" ]; then
    CURL_ARGS+=(-H "Authorization: $CLAUDE_WEBHOOK_AUTH")
fi

# Send request
RESPONSE=$(curl "${CURL_ARGS[@]}" --data "$INPUT" "$CLAUDE_WEBHOOK_URL" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[WEBHOOK] Notification sent to $CLAUDE_WEBHOOK_URL"
    echo "[$TIMESTAMP] [SENT] $INPUT -> $RESPONSE" >> "$LOG_FILE"
else
    echo "[WEBHOOK] Failed to send notification: $RESPONSE" >&2
    echo "[$TIMESTAMP] [FAILED] $INPUT -> $RESPONSE" >> "$LOG_FILE"
fi

exit 0
