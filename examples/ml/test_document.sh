#!/bin/bash
# Test Document Understanding endpoint with image upload
#
# This script demonstrates:
# 1. Uploading a receipt image
# 2. Running document VQA using Donut DocVQA model
# 3. Cleaning up the file
#
# Usage: ./test_document.sh [PORT] [IMAGE_FILE]
#   PORT defaults to 11540 (Universal Runtime)
#   IMAGE_FILE defaults to the sample receipt in this directory

set -e

PORT=${1:-11540}
IMAGE_FILE=${2:-"$(dirname "$0")/receipt.png"}
BASE_URL="http://localhost:${PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Universal Runtime Document Understanding Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Check if file exists
if [ ! -f "$IMAGE_FILE" ]; then
    echo -e "${RED}Error: File not found: ${IMAGE_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}1. Uploading receipt image...${NC}"
echo "   File: $(basename "$IMAGE_FILE")"
echo ""

# Upload the file
UPLOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/files" \
    -F "file=@${IMAGE_FILE}")

echo "Upload Response:"
echo "$UPLOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$UPLOAD_RESPONSE"
echo ""

# Extract file_id from response
FILE_ID=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null)

if [ -z "$FILE_ID" ]; then
    echo -e "${RED}Error: Failed to get file_id from upload response${NC}"
    exit 1
fi

echo -e "${GREEN}✓ File uploaded: ${FILE_ID}${NC}"
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

    DOC_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/documents/extract" \
        -H "Content-Type: application/json" \
        --max-time 300 \
        -d "{
            \"model\": \"naver-clova-ix/donut-base-finetuned-docvqa\",
            \"file_id\": \"${FILE_ID}\",
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

# Clean up - delete the uploaded file
echo -e "${YELLOW}3. Cleaning up uploaded file...${NC}"
DELETE_RESPONSE=$(curl -s -X DELETE "${BASE_URL}/v1/files/${FILE_ID}")
echo "$DELETE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DELETE_RESPONSE"
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
