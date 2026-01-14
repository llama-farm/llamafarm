#!/bin/bash
# Demo script for dynamic tool variables
#
# This script demonstrates using the `variables` field to customize
# tool definitions at runtime.
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - Universal runtime running with a model available
#
# Usage:
#   bash examples/dynamic_values_tools/demo.sh

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

echo "=== Dynamic Tool Variables Demo ==="
echo "API: $API_BASE"
echo "Project: $NAMESPACE/$PROJECT"
echo ""

# Test 1: Tool with custom API base
echo "Test 1: Tool with custom API variables"
echo "---------------------------------------"
echo "Setting api_name='ProductService' and api_base='https://api.products.com'"
echo ""

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "What tools do you have available?"}],
    "variables": {
      "api_name": "ProductService",
      "api_base": "https://api.products.com"
    },
    "max_tokens": 200
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention ProductService and api.products.com): "
  echo "$content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

# Test 2: Tool using defaults
echo "Test 2: Tool using defaults (no variables)"
echo "-------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Describe your fetch_data tool."}],
    "max_tokens": 200
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention 'the API' and localhost:3000 defaults): "
  echo "$content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

# Test 3: Request-level tool with variables
echo "Test 3: Request-level tool with variables"
echo "------------------------------------------"

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "What can you do?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "custom_tool",
          "description": "A custom tool for {{service_name | MyService}}",
          "parameters": {
            "type": "object",
            "properties": {
              "input": {"type": "string"}
            }
          }
        }
      }
    ],
    "variables": {
      "service_name": "CustomerSupport"
    },
    "max_tokens": 200
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention CustomerSupport): "
  echo "$content"
else
  echo "Error or unexpected response: $response"
fi
echo ""

echo "=== Demo complete ==="
echo ""
echo "Key takeaways:"
echo "1. Tool descriptions can use {{variable}} syntax"
echo "2. Tool parameter descriptions/defaults can use variables"
echo "3. Both config tools and request tools are resolved"
echo "4. Defaults work: {{api_base | http://localhost:3000}}"
