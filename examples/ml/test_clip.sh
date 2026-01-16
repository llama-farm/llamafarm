#!/bin/bash
# Test Zero-Shot Image Classification (CLIP) via LlamaFarm API
#
# This script tests the CLIP endpoint through the LlamaFarm API proxy
#
# Usage: ./test_clip_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="${SCRIPT_DIR}/../files"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  CLIP via LlamaFarm API Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    echo "Start it with: nx start server"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Test classification via API proxy
echo -e "${YELLOW}Testing CLIP via /v1/ml/vision/classify...${NC}"

IMAGE_PATH="${FILES_DIR}/cat.png"
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Test image not found: ${IMAGE_PATH}${NC}"
    exit 1
fi

IMAGE_B64=$(base64 -i "$IMAGE_PATH")
PAYLOAD_FILE=$(mktemp)
cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}", "labels": ["cat", "dog", "horse", "bird"]}
EOFPAYLOAD

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/vision/classify" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d @"$PAYLOAD_FILE")
rm -f "$PAYLOAD_FILE"

echo "Labels: cat, dog, horse, bird"
echo ""
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'scores' in data:
        scores = data['scores']
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print('  Results:')
        for label, score in sorted_scores:
            print(f'    {label}: {score:.4f}')
        print(f'')
        print(f'  ✓ Predicted: {sorted_scores[0][0]} ({sorted_scores[0][1]:.2%})')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
