#!/bin/bash
# Test OCR endpoint via LlamaFarm API (proxied to Universal Runtime)
#
# This script demonstrates running OCR by uploading files directly
# to the /v1/vision/ocr endpoint.
#
# Usage: ./test_ocr_api.sh [PORT] [IMAGE_FILE]
#   PORT defaults to 14345 (LlamaFarm API)
#   IMAGE_FILE defaults to the sample image in this directory

set -e

PORT=${1:-14345}
IMAGE_FILE=${2:-"$(dirname "$0")/receipt.png"}
BASE_URL="http://localhost:${PORT}/v1/vision"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  LlamaFarm API OCR Test (via /v1/vision)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    echo "Start it with: nx start server"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Check Universal Runtime health via ML proxy (health is still on /v1/ml)
echo -e "${YELLOW}Checking Universal Runtime health...${NC}"
if ! curl -sf "http://localhost:${PORT}/v1/ml/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not available${NC}"
    echo "Start it with: nx start universal"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# Check if file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}Error: File not found: ${IMAGE_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}1. Uploading file for OCR...${NC}"
echo "   File: $(basename "$IMAGE_FILE")"
echo ""

echo -e "${YELLOW}2. Running OCR with EasyOCR backend...${NC}"
echo "   (EasyOCR is widely available and doesn't require GPU)"
echo ""

# Run OCR with file upload
OCR_RESPONSE=$(curl -s -X POST "${BASE_URL}/ocr" \
    -F "file=@${IMAGE_FILE}" \
    -F "model=easyocr" \
    -F "languages=en" \
    -F "return_boxes=false")

echo "OCR Response:"
echo "$OCR_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$OCR_RESPONSE"
echo ""

# Check if OCR was successful
if echo "$OCR_RESPONSE" | grep -q '"text"'; then
    echo -e "${GREEN}✓ OCR completed successfully!${NC}"

    # Extract just the text
    echo ""
    echo -e "${BLUE}Extracted Text:${NC}"
    echo "---"
    echo "$OCR_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for item in data.get('data', []):
        print(f\"Image {item['index'] + 1}:\")
        print(item.get('text', 'No text found')[:500])
        print('...' if len(item.get('text', '')) > 500 else '')
        print()
except Exception as e:
    print(f'Error parsing response: {e}')
" 2>/dev/null
    echo "---"
else
    echo -e "${YELLOW}Note: OCR may have failed or returned empty results${NC}"
    echo "This could be because:"
    echo "  - The OCR backend (easyocr) is not installed"
    echo "  - The image quality is too low"
    echo "  - The document contains no recognizable text"
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
