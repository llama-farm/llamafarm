#!/bin/bash
# Demo script for ChatRequest with variables field
#
# This script demonstrates that the LlamaFarm API accepts the new
# `variables` field in chat completion requests.
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - A project configured (uses default/project_seed as example)
#
# Usage:
#   bash examples/dynamic_values_basic/test_api_request.sh

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

echo "=== Testing ChatRequest with variables field ==="
echo "API: $API_BASE"
echo "Project: $NAMESPACE/$PROJECT"
echo ""

# Test 1: Request with variables
echo "Test 1: Request with variables field"
echo "--------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello"}],
    "variables": {
      "user_name": "Alice",
      "company": "Acme Corp",
      "tier": "premium"
    },
    "max_tokens": 50
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  echo "SUCCESS: Request with variables accepted"
  echo "Response preview: $(echo "$response" | head -c 200)..."
else
  echo "Response: $response"
  # Check if it's a validation error (which would mean the field exists but something else is wrong)
  if echo "$response" | grep -q "validation"; then
    echo "Note: Validation error suggests field exists but request has other issues"
  fi
fi
echo ""

# Test 2: Request without variables (backwards compatibility)
echo "Test 2: Request without variables (backwards compatibility)"
echo "------------------------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 50
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  echo "SUCCESS: Request without variables still works"
  echo "Response preview: $(echo "$response" | head -c 200)..."
else
  echo "Response: $response"
fi
echo ""

# Test 3: Check OpenAPI schema includes variables
echo "Test 3: Check OpenAPI schema for variables field"
echo "-------------------------------------------------"

schema=$(curl -s "$API_BASE/openapi.json" 2>&1)

if echo "$schema" | grep -q '"variables"'; then
  echo "SUCCESS: OpenAPI schema includes 'variables' field"
else
  echo "Note: Could not verify OpenAPI schema (server may need restart)"
fi
echo ""

echo "=== Demo complete ==="
