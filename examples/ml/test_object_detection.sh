#!/bin/bash
# Test Object Detection (YOLOS) endpoint
#
# This script demonstrates:
# 1. Detecting objects in images
# 2. Processing multiple images
# 3. Showing bounding boxes and confidence scores
#
# Usage: ./test_object_detection.sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (fallback: 11540)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi

PORT=${1:-${LF_RUNTIME_PORT:-11540}}
BASE_URL="http://localhost:${PORT}"
FILES_DIR="${SCRIPT_DIR}/../files"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Object Detection (YOLOS) Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# ============================================================================
# Test 1: Detect Objects in Single Image
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Detect Objects in Single Image${NC}"
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
{"image": "${IMAGE_B64}", "threshold": 0.5}
EOFPAYLOAD

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/detect" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d @"$PAYLOAD_FILE")
rm -f "$PAYLOAD_FILE"

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'detections' in data:
        detections = data['detections']
        if detections:
            print(f'  Found {len(detections)} object(s):')
            for d in detections:
                label = d.get('label', 'unknown')
                score = d.get('score', 0)
                box = d.get('box', {})
                print(f'    - {label}: {score:.2%} confidence')
                if box:
                    print(f'      Box: x={box.get(\"x\", 0):.0f}, y={box.get(\"y\", 0):.0f}, w={box.get(\"width\", 0):.0f}, h={box.get(\"height\", 0):.0f}')
        else:
            print('  No objects detected above threshold')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Detect Objects in Multiple Images
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Detect Objects in Multiple Images${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

for IMG in cat1.jpg cat2.jpg horse.jpg; do
    IMAGE_PATH="${FILES_DIR}/${IMG}"
    if [ -f "$IMAGE_PATH" ]; then
        echo -e "${YELLOW}Testing with ${IMG}...${NC}"
        IMAGE_B64=$(base64 -i "$IMAGE_PATH")
        PAYLOAD_FILE=$(mktemp)
        cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}", "threshold": 0.3}
EOFPAYLOAD

        RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/detect" \
            -H "Content-Type: application/json" \
            --max-time 120 \
            -d @"$PAYLOAD_FILE")
        rm -f "$PAYLOAD_FILE"

        echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'detections' in data:
        detections = data['detections']
        if detections:
            labels = [f\"{d['label']} ({d['score']:.0%})\" for d in detections[:3]]
            print(f'  ✓ Detected: {', '.join(labels)}')
        else:
            print('  No objects detected')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
        echo ""
    fi
done

# ============================================================================
# Test 3: Threshold Comparison
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Threshold Comparison${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Testing different confidence thresholds on cat.png...${NC}"
IMAGE_PATH="${FILES_DIR}/cat.png"
IMAGE_B64=$(base64 -i "$IMAGE_PATH")

for THRESHOLD in 0.9 0.5 0.3 0.1; do
    PAYLOAD_FILE=$(mktemp)
    cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}", "threshold": ${THRESHOLD}}
EOFPAYLOAD
    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/detect" \
        -H "Content-Type: application/json" \
        --max-time 120 \
        -d @"$PAYLOAD_FILE")
    rm -f "$PAYLOAD_FILE"

    COUNT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(len(data.get('detections', [])))
except:
    print(0)
" 2>/dev/null)

    echo "  Threshold ${THRESHOLD}: ${COUNT} detection(s)"
done
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoint: POST /v1/vision/detect"
echo ""
echo "Request format:"
echo "  {"
echo "    \"image\": \"<base64 encoded image>\","
echo "    \"threshold\": 0.5  // Optional, default 0.5"
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"detections\": ["
echo "      {"
echo "        \"label\": \"cat\","
echo "        \"score\": 0.95,"
echo "        \"box\": {\"x\": 10, \"y\": 20, \"width\": 100, \"height\": 150}"
echo "      }"
echo "    ]"
echo "  }"
echo ""
