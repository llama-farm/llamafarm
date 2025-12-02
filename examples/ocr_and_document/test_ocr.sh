#!/bin/bash
# Test OCR endpoint with file upload
#
# This script demonstrates:
# 1. Uploading a PDF file (auto-converts to images)
# 2. Running OCR on the uploaded file
# 3. Cleaning up the file
#
# Usage: ./test_ocr.sh [PORT] [PDF_FILE]
#   PORT defaults to 11540 (Universal Runtime)
#   PDF_FILE defaults to the sample PDF in this directory

set -e

PORT=${1:-11540}
PDF_FILE=${2:-"$(dirname "$0")/llamafarm - Healthcare - Aug 2025 2 .pdf"}
BASE_URL="http://localhost:${PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Universal Runtime OCR Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Check if file exists
if [ ! -f "$PDF_FILE" ]; then
    echo -e "${RED}Error: File not found: ${PDF_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}1. Uploading PDF file...${NC}"
echo "   File: $(basename "$PDF_FILE")"
echo ""

# Upload the file
UPLOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/files" \
    -F "file=@${PDF_FILE}" \
    -F "convert_pdf=true" \
    -F "pdf_dpi=150")

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

# Get file images info
echo -e "${YELLOW}2. Getting file images info...${NC}"
curl -s "${BASE_URL}/v1/files/${FILE_ID}" | python3 -m json.tool 2>/dev/null
echo ""

echo -e "${YELLOW}3. Running OCR with EasyOCR backend...${NC}"
echo "   (EasyOCR is widely available and doesn't require GPU)"
echo ""

# Run OCR using file_id
OCR_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ocr" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"easyocr\",
        \"file_id\": \"${FILE_ID}\",
        \"languages\": [\"en\"],
        \"return_boxes\": false
    }")

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
        print(f\"Page {item['index'] + 1}:\")
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

# Clean up - delete the uploaded file
echo -e "${YELLOW}4. Cleaning up uploaded file...${NC}"
DELETE_RESPONSE=$(curl -s -X DELETE "${BASE_URL}/v1/files/${FILE_ID}")
echo "$DELETE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DELETE_RESPONSE"
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
