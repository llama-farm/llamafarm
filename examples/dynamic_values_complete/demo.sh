#!/bin/bash
# Complete demo of dynamic values in LlamaFarm
#
# This script demonstrates:
# 1. Dynamic prompts with {{variable | default}} syntax
# 2. Dynamic tool definitions with variable descriptions
# 3. Different variables per request
# 4. Defaults falling back when variables not provided
# 5. Mixing prompts and tools together
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - Universal runtime with llama3.2:3b available
#
# Usage:
#   bash examples/dynamic_values_complete/demo.sh

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

echo "=============================================="
echo "  LlamaFarm Dynamic Values - Complete Demo"
echo "=============================================="
echo ""
echo "API: $API_BASE"
echo "Project: $NAMESPACE/$PROJECT"
echo ""

# Scenario 1: Premium customer with full variables
echo "=============================================="
echo "Scenario 1: Premium Customer (Full Variables)"
echo "=============================================="
echo ""
echo "Variables:"
echo "  - company_name: TechCorp Solutions"
echo "  - department: Technical Support"
echo "  - user_name: Alice Johnson"
echo "  - account_tier: premium"
echo "  - current_date: 2024-01-15"
echo ""

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, can you tell me about yourself and what you can help me with?"}],
    "variables": {
      "company_name": "TechCorp Solutions",
      "department": "Technical Support",
      "user_name": "Alice Johnson",
      "account_tier": "premium",
      "current_date": "2024-01-15",
      "language": "English"
    },
    "max_tokens": 300
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response:"
  echo "$content"
else
  echo "Error: $response"
fi
echo ""

# Scenario 2: Basic customer using mostly defaults
echo "=============================================="
echo "Scenario 2: Basic Customer (Using Defaults)"
echo "=============================================="
echo ""
echo "Variables: Only user_name and account_tier provided"
echo "Others use defaults: company=Acme Corp, department=General, etc."
echo ""

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "What can you help me with?"}],
    "variables": {
      "user_name": "Bob",
      "account_tier": "basic"
    },
    "max_tokens": 200
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should mention self-service for basic tier):"
  echo "$content"
else
  echo "Error: $response"
fi
echo ""

# Scenario 3: No variables at all (all defaults)
echo "=============================================="
echo "Scenario 3: Anonymous User (All Defaults)"
echo "=============================================="
echo ""
echo "Variables: None provided - all defaults used"
echo "Expected: Acme Corp, General department, Valued Customer, standard tier"
echo ""

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 150
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response:"
  echo "$content"
else
  echo "Error: $response"
fi
echo ""

# Scenario 4: With additional request tool
echo "=============================================="
echo "Scenario 4: Custom Request Tool with Variables"
echo "=============================================="
echo ""
echo "Adding a custom tool that also uses variables"
echo ""

response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d '{
    "messages": [{"role": "user", "content": "What tools do you have?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "custom_lookup",
          "description": "Look up information in {{custom_service | the external system}}",
          "parameters": {
            "type": "object",
            "properties": {
              "id": {"type": "string", "description": "ID to look up"}
            },
            "required": ["id"]
          }
        }
      }
    ],
    "variables": {
      "company_name": "Demo Corp",
      "custom_service": "the CRM database",
      "user_name": "Demo User"
    },
    "max_tokens": 250
  }' 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response (should list tools including custom_lookup for CRM):"
  echo "$content"
else
  echo "Error: $response"
fi
echo ""

echo "=============================================="
echo "                Demo Complete"
echo "=============================================="
echo ""
echo "Summary of Dynamic Values Features:"
echo ""
echo "1. SYNTAX"
echo "   {{variable}}              - Required variable"
echo "   {{variable | default}}    - Variable with default"
echo "   {{ variable }}            - Whitespace is OK"
echo ""
echo "2. WHERE TO USE"
echo "   - Prompt message content"
echo "   - Tool descriptions"
echo "   - Tool parameter descriptions"
echo "   - Tool parameter defaults"
echo ""
echo "3. HOW TO PASS"
echo "   Add 'variables' field to your request:"
echo '   {"messages": [...], "variables": {"key": "value"}}'
echo ""
echo "4. BACKWARDS COMPATIBLE"
echo "   - Existing configs without {{}} work unchanged"
echo "   - 'variables' field is optional"
echo ""
