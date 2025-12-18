#!/bin/bash
# Test Document Understanding endpoint via LlamaFarm API
#
# This script demonstrates running document VQA by passing base64-encoded images
# directly to the /v1/vision/documents/extract endpoint.
#
# Usage: ./test_document_api.sh [PORT] [IMAGE_FILE]
#   PORT defaults to 8000 (LlamaFarm API)
#   IMAGE_FILE defaults to the sample receipt in this directory

set -e

PORT=${1:-8000}
IMAGE_FILE=${2:-"$(dirname "$0")/receipt.png"}
BASE_URL="http://localhost:${PORT}/v1/vision"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  LlamaFarm API Document Understanding Test${NC}"
echo -e "${BLUE}  (via /v1/vision)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if LlamaFarm server is running
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    echo "Start it with: nx start server"
    exit 1
fi
echo -e "${GREEN}✓ LlamaFarm API is healthy${NC}"

# Check Universal Runtime via ML health endpoint
echo -e "${YELLOW}Checking Universal Runtime...${NC}"
if ! curl -s "http://localhost:${PORT}/v1/ml/health" > /dev/null 2>&1; then
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

echo -e "${YELLOW}1. Encoding image to base64...${NC}"
echo "   File: $(basename "$IMAGE_FILE")"
echo ""

# Encode image to base64
BASE64_IMAGE=$(base64 < "$IMAGE_FILE" | tr -d '\n')
MIME_TYPE="image/png"
if [[ "$IMAGE_FILE" == *.jpg ]] || [[ "$IMAGE_FILE" == *.jpeg ]]; then
    MIME_TYPE="image/jpeg"
fi

echo -e "${GREEN}✓ Image encoded (${#BASE64_IMAGE} chars)${NC}"
echo ""

echo -e "${YELLOW}2. Running Document VQA with Donut DocVQA...${NC}"
echo "   Model: naver-clova-ix/donut-base-finetuned-docvqa"
echo "   Task: Visual Question Answering"
echo ""
echo -e "${YELLOW}   Note: First run downloads the model (~1GB)...${NC}"
echo ""

# Questions to ask about the receipt
QUESTIONS=(
    "What is the store name?"
    "What is the total amount?"
    "What items were purchased?"
    "What is the date?"
)

for QUESTION in "${QUESTIONS[@]}"; do
    echo -e "${BLUE}Question: ${QUESTION}${NC}"

    DOC_RESPONSE=$(curl -s -X POST "${BASE_URL}/documents/extract" \
        -H "Content-Type: application/json" \
        --max-time 300 \
        -d "{
            \"model\": \"naver-clova-ix/donut-base-finetuned-docvqa\",
            \"images\": [\"data:${MIME_TYPE};base64,${BASE64_IMAGE}\"],
            \"prompts\": [\"${QUESTION}\"],
            \"task\": \"vqa\"
        }")

    # Extract answer
    ANSWER=$(echo "$DOC_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for item in data.get('data', []):
        if 'text' in item:
            # Parse out the answer from Donut's format
            text = item['text']
            # Remove special tokens
            text = text.replace('<s_docvqa>', '').replace('</s_docvqa>', '')
            text = text.replace('<s_answer>', '').replace('</s_answer>', '')
            text = text.replace('<s_question>', '').replace('</s_question>', '')
            text = text.strip()
            if text:
                print(f'  Answer: {text}')
            else:
                print('  Answer: (no answer extracted)')
        elif 'answer' in item:
            print(f'  Answer: {item[\"answer\"]}')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null || echo "  (parsing error)")

    echo "$ANSWER"
    echo ""
done

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
