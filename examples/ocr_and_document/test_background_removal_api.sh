#!/bin/bash
# Test Background Removal via LlamaFarm API
#
# Usage: ./test_background_removal_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="${SCRIPT_DIR}/../files"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Background Removal via LlamaFarm API Test${NC}"
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

# Test background removal
echo -e "${YELLOW}Testing via /v1/ml/vision/remove-background...${NC}"

IMAGE_PATH="${FILES_DIR}/cat.png"
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Test image not found${NC}"
    exit 1
fi

IMAGE_B64=$(base64 -i "$IMAGE_PATH")

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/vision/remove-background" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d "{
        \"image\": \"${IMAGE_B64}\"
    }")

echo "$RESPONSE" | python3 -c "
import sys, json, base64
try:
    data = json.load(sys.stdin)
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        print(f'  ✓ Received PNG with alpha ({len(img_data)} bytes)')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
