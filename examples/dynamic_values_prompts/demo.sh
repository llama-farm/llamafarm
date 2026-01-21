#!/bin/bash
# Demo script for dynamic prompt variables
#
# This script demonstrates using the `variables` field to customize
# prompts at runtime without changing the config.
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - Universal runtime running with llama3.2:3b model available
#   - This example project registered (or use default/project_seed)
#
# Usage:
#   bash examples/dynamic_values_prompts/demo.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
  PORT=$(grep -E "^PORT=" "$ENV_FILE" | cut -d= -f2)
fi
PORT="${PORT:-8000}"

API_BASE="${API_BASE:-http://localhost:$PORT}"
NAMESPACE="${NAMESPACE:-test}"
PROJECT="${PROJECT:-demo}"

echo "=== Dynamic Prompt Variables Demo ==="
echo "API: $API_BASE"
echo "Project: $NAMESPACE/$PROJECT"
echo ""

# Test 1: With custom variables
echo "Test 1: Chat with custom variables (user_name=Alice, company_name=TechCorp)"
echo "------------------------------------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Introduce yourself briefly."}],
    "variables": {
      "user_name": "Alice",
      "company_name": "TechCorp",
      "department": "Engineering",
      "role_description": "technical support and code reviews"
    },
    "max_tokens": 100
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response: $content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

# Test 2: Using defaults (no variables)
echo "Test 2: Chat using defaults (no variables provided)"
echo "----------------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Who are you?"}],
    "max_tokens": 100
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention 'Guest', 'Our Service', 'General'): $content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

# Test 3: Partial variables (mix of provided and defaults)
echo "Test 3: Partial variables (only user_name provided)"
echo "----------------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "What department do you work for?"}],
    "variables": {
      "user_name": "Bob"
    },
    "max_tokens": 100
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention 'General' department from default): $content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

echo "=== Demo complete ==="
echo ""
echo "Key takeaways:"
echo "1. Use {{variable_name}} syntax in your prompts"
echo "2. Use {{variable_name | default}} for optional variables"
echo "3. Pass variables via the 'variables' field in the request"
echo "4. Existing prompts without variables work unchanged (backwards compatible)"
