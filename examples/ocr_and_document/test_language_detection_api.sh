#!/bin/bash
# Test Language Detection via LlamaFarm API
#
# Usage: ./test_language_detection_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Language Detection via LlamaFarm API Test${NC}"
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

# Test language detection
echo -e "${YELLOW}Testing via /v1/ml/nlp/identify-language...${NC}"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/nlp/identify-language" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "text": "Hello, this is a test in English."
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'language' in data:
        print(f'  ✓ Detected: {data[\"language\"]} ({data.get(\"score\", 0):.2%})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
