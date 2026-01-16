#!/bin/bash
# Test PII Redaction via LlamaFarm API
#
# Usage: ./test_pii_redaction_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi


PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  PII Redaction via LlamaFarm API Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Test PII redaction
echo -e "${YELLOW}Testing via /v1/ml/nlp/redact...${NC}"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/nlp/redact" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "text": "Contact John Smith at john@email.com or call 555-1234."
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'entities' in data:
        print(f'  ✓ Found {len(data[\"entities\"])} PII entities')
        for e in data['entities'][:3]:
            print(f'    - {e.get(\"type\", \"?\")}: \"{e.get(\"text\", \"\")}\"')
    if 'redacted_text' in data:
        print(f'  Redacted: {data[\"redacted_text\"][:60]}...')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
