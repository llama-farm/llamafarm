#!/bin/bash
#
# Slack Handler - Receives Slack commands and runs Claude headlessly
#
# This script is designed to be called by a webhook receiver (e.g., via ngrok)
# It parses Slack slash commands and invokes Claude with the appropriate prompt.
#
# Usage:
#   echo '{"text": "build a user API", "user_name": "rob"}' | ./slack-handler.sh
#
# Environment variables:
#   CLAUDE_PROJECT_DIR - Project directory to work in
#   CLAUDE_SLACK_WEBHOOK - Webhook URL for status updates
#

set -e

# Get project directory
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
SCRIPT_DIR="$(dirname "$0")"

# Read JSON input from stdin
INPUT=$(cat)

# Parse Slack command payload
TEXT=$(echo "$INPUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('text', ''))" 2>/dev/null || echo "")
USER=$(echo "$INPUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('user_name', 'unknown'))" 2>/dev/null || echo "unknown")
CHANNEL=$(echo "$INPUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('channel_name', 'unknown'))" 2>/dev/null || echo "unknown")

if [ -z "$TEXT" ]; then
    echo '{"response_type": "ephemeral", "text": "Please provide a task. Usage: /claude [task description]"}'
    exit 0
fi

# Log the request
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "[$TIMESTAMP] Received from $USER in #$CHANNEL: $TEXT" >> "$PROJECT_DIR/.claude/context/slack-requests.log"

# Send immediate acknowledgment (Slack requires response within 3 seconds)
echo '{"response_type": "in_channel", "text": "🚀 Starting task: '"${TEXT:0:100}"'..."}'

# Detect template based on keywords
TEMPLATE=""
if echo "$TEXT" | grep -qi "feature\|add\|create\|build\|implement"; then
    TEMPLATE="new-feature"
elif echo "$TEXT" | grep -qi "fix\|bug\|error\|issue"; then
    TEMPLATE="bug-fix"
elif echo "$TEXT" | grep -qi "llamafarm\|lf\|rag\|ml"; then
    TEMPLATE="llamafarm-app"
fi

# Run Claude in background
(
    # Send initial notification
    if [ -n "$CLAUDE_SLACK_WEBHOOK" ]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\": \"📋 *Task from $USER*: $TEXT\", \"channel\": \"#$CHANNEL\"}" \
            "$CLAUDE_SLACK_WEBHOOK" > /dev/null
    fi

    # Run headless Claude
    if [ -n "$TEMPLATE" ]; then
        "$SCRIPT_DIR/run-headless.sh" --template "$TEMPLATE" "$TEXT" "$PROJECT_DIR"
    else
        "$SCRIPT_DIR/run-headless.sh" "$TEXT" "$PROJECT_DIR"
    fi

    # Send completion notification
    if [ -n "$CLAUDE_SLACK_WEBHOOK" ]; then
        RESULT=$(cat "$PROJECT_DIR/.claude/context/last-run.json" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('result', 'Task completed')[:500])
except:
    print('Task completed')
" 2>/dev/null || echo "Task completed")

        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\": \"✅ *Task Complete*\n\n$RESULT\"}" \
            "$CLAUDE_SLACK_WEBHOOK" > /dev/null
    fi
) &

# Return immediately to Slack
exit 0
