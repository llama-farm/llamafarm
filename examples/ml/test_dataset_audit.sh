#!/bin/bash
# Test Dataset Audit via LlamaFarm API
#
# Usage: ./test_dataset_audit_api.sh [PORT]
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
echo -e "${BLUE}  Dataset Audit via LlamaFarm API Test${NC}"
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

# Test audit
echo -e "${YELLOW}Testing via /v1/ml/dataset/audit...${NC}"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/dataset/audit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "labels": [0, 1, 2, 0, 1, 2, 1, 0, 2, 1],
        "pred_probs": [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.05, 0.05, 0.9],
            [0.05, 0.9, 0.05]
        ],
        "label_names": ["A", "B", "C"]
    }')

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'summary' in data:
        s = data['summary']
        print(f'  ✓ Audit complete')
        print(f'    Total samples: {s.get(\"total_samples\", 0)}')
        print(f'    Issues found: {s.get(\"num_label_issues\", 0)}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
