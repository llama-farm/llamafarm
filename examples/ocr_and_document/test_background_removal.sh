#!/bin/bash
# Test Background Removal (RMBG) endpoint
#
# This script demonstrates:
# 1. Removing background from images
# 2. Saving results with alpha channel
#
# Usage: ./test_background_removal.sh [PORT]
#   PORT defaults to 11540 (Universal Runtime)

set -e

PORT=${1:-11540}
BASE_URL="http://localhost:${PORT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="${SCRIPT_DIR}/../files"
OUTPUT_DIR="/tmp/rmbg_test_output"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Background Removal (RMBG) Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Test 1: Remove Background from Single Image
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Remove Background${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Testing with cat.png...${NC}"
IMAGE_PATH="${FILES_DIR}/cat.png"
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Test image not found: ${IMAGE_PATH}${NC}"
    exit 1
fi

IMAGE_B64=$(base64 -i "$IMAGE_PATH")
PAYLOAD_FILE=$(mktemp)
cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}"}
EOFPAYLOAD

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/remove-background" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d @"$PAYLOAD_FILE")
rm -f "$PAYLOAD_FILE"

# Save the result
echo "$RESPONSE" | python3 -c "
import sys, json, base64
try:
    data = json.load(sys.stdin)
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        output_path = '${OUTPUT_DIR}/cat_no_bg.png'
        with open(output_path, 'wb') as f:
            f.write(img_data)
        print(f'  ✓ Saved to: {output_path}')
        print(f'  Size: {len(img_data)} bytes')

        # Verify it's a PNG with alpha
        if img_data[:8] == b'\\x89PNG\\r\\n\\x1a\\n':
            print('  Format: PNG ✓')
        else:
            print('  Format: Unknown')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Process Multiple Images
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Process Multiple Images${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

for IMG in cat1.jpg horse.jpg; do
    IMAGE_PATH="${FILES_DIR}/${IMG}"
    if [ -f "$IMAGE_PATH" ]; then
        echo -e "${YELLOW}Processing ${IMG}...${NC}"
        IMAGE_B64=$(base64 -i "$IMAGE_PATH")
        PAYLOAD_FILE=$(mktemp)
        cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}"}
EOFPAYLOAD

        RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/remove-background" \
            -H "Content-Type: application/json" \
            --max-time 120 \
            -d @"$PAYLOAD_FILE")
        rm -f "$PAYLOAD_FILE"

        OUTPUT_NAME="${IMG%.*}_no_bg.png"
        echo "$RESPONSE" | python3 -c "
import sys, json, base64
output_name = '${OUTPUT_NAME}'
try:
    data = json.load(sys.stdin)
    if 'image' in data:
        img_data = base64.b64decode(data['image'])
        output_path = f'${OUTPUT_DIR}/{output_name}'
        with open(output_path, 'wb') as f:
            f.write(img_data)
        print(f'  ✓ Saved: {output_name} ({len(img_data)} bytes)')
    else:
        print(f'  Error: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
        echo ""
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Output files saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>/dev/null || echo "  (directory is empty)"
echo ""
echo "API Endpoint: POST /v1/vision/remove-background"
echo ""
echo "Request format:"
echo "  {"
echo "    \"image\": \"<base64 encoded image>\""
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"image\": \"<base64 encoded PNG with alpha>\""
echo "  }"
echo ""
