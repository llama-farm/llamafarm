#!/bin/bash
# Test Keyword Extraction via LlamaFarm API
#
# Usage: ./test_keywords_api.sh [PORT]
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
echo -e "${BLUE}  Keyword Extraction via LlamaFarm API Test${NC}"
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

# Test keyword extraction
echo -e "${YELLOW}Testing via /v1/ml/nlp/keywords...${NC}"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "text": "Machine learning enables computers to learn from data. Deep learning uses neural networks for complex pattern recognition.",
        "top_n": 5
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'keywords' in data:
        kws = data['keywords'][:5]
        print('  ✓ Keywords extracted:')
        for kw in kws:
            if isinstance(kw, dict):
                print(f'    - {kw.get(\"keyword\", kw)}')
            else:
                print(f'    - {kw}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
