#!/bin/bash
# Test Zero-Shot Image Classification (CLIP) endpoint
#
# This script demonstrates:
# 1. Zero-shot image classification with custom labels
# 2. Processing multiple images
# 3. Processing video frames
#
# Usage: ./test_clip.sh [PORT]
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Zero-Shot Image Classification (CLIP) Test${NC}"
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
# Test 1: Classify Single Image
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Classify Single Image${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Testing with cat.png...${NC}"
IMAGE_PATH="${FILES_DIR}/cat.png"
if [ ! -f "$IMAGE_PATH" ]; then
    echo -e "${RED}Error: Test image not found: ${IMAGE_PATH}${NC}"
    exit 1
fi

# Base64 encode the image and create JSON payload file
IMAGE_B64=$(base64 -i "$IMAGE_PATH")
PAYLOAD_FILE=$(mktemp)
cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{
    "image": "${IMAGE_B64}",
    "labels": ["cat", "dog", "horse", "bird", "fish", "bear"]
}
EOFPAYLOAD

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/classify" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d @"$PAYLOAD_FILE")
rm -f "$PAYLOAD_FILE"

echo "Labels: cat, dog, horse, bird, fish, bear"
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
            bar = '█' * int(score * 20)
            print(f'    {label:<10}: {score:.4f} {bar}')
        print(f'')
        print(f'  ✓ Predicted: {sorted_scores[0][0]} ({sorted_scores[0][1]:.2%})')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Classify Multiple Images
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Classify Multiple Images${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

LABELS='["cat", "dog", "horse", "bird", "fish", "bear"]'

for IMG in cat1.jpg cat2.jpg horse.jpg; do
    IMAGE_PATH="${FILES_DIR}/${IMG}"
    if [ -f "$IMAGE_PATH" ]; then
        echo -e "${YELLOW}Testing with ${IMG}...${NC}"
        IMAGE_B64=$(base64 -i "$IMAGE_PATH")
        PAYLOAD_FILE=$(mktemp)
        cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}", "labels": ["cat", "dog", "horse", "bird", "fish", "bear"]}
EOFPAYLOAD

        RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/classify" \
            -H "Content-Type: application/json" \
            --max-time 120 \
            -d @"$PAYLOAD_FILE")
        rm -f "$PAYLOAD_FILE"

        echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'scores' in data:
        scores = data['scores']
        top = max(scores.items(), key=lambda x: x[1])
        print(f'  ✓ Predicted: {top[0]} ({top[1]:.2%})')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
        echo ""
    fi
done

# ============================================================================
# Test 3: Video Frame Classification
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Video Frame Classification${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

VIDEO_PATH="${FILES_DIR}/bird.mp4"
if [ -f "$VIDEO_PATH" ] && command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Extracting frames from bird.mp4...${NC}"

    # Create temp directory for frames
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT

    # Extract frames every 1 second
    ffmpeg -i "$VIDEO_PATH" -vf "fps=1" -q:v 2 "$TEMP_DIR/frame_%03d.jpg" 2>/dev/null

    FRAME_COUNT=$(ls "$TEMP_DIR"/frame_*.jpg 2>/dev/null | wc -l)
    echo "  Extracted ${FRAME_COUNT} frames"
    echo ""

    echo -e "${YELLOW}Classifying frames...${NC}"
    FRAME_NUM=0
    for FRAME in "$TEMP_DIR"/frame_*.jpg; do
        if [ -f "$FRAME" ]; then
            FRAME_NUM=$((FRAME_NUM + 1))
            IMAGE_B64=$(base64 -i "$FRAME")
            PAYLOAD_FILE=$(mktemp)
            cat > "$PAYLOAD_FILE" << EOFPAYLOAD
{"image": "${IMAGE_B64}", "labels": ["cat", "dog", "horse", "bird", "fish", "bear"]}
EOFPAYLOAD

            RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/vision/classify" \
                -H "Content-Type: application/json" \
                --max-time 120 \
                -d @"$PAYLOAD_FILE")
            rm -f "$PAYLOAD_FILE"

            echo "$RESPONSE" | python3 -c "
import sys, json
frame_num = ${FRAME_NUM}
try:
    data = json.load(sys.stdin)
    if 'scores' in data:
        scores = data['scores']
        top = max(scores.items(), key=lambda x: x[1])
        print(f'  Frame {frame_num}: {top[0]} ({top[1]:.2%})')
except:
    pass
" 2>/dev/null

            # Limit to 5 frames for demo
            if [ $FRAME_NUM -ge 5 ]; then
                echo "  ... (showing first 5 frames)"
                break
            fi
        fi
    done
else
    if [ ! -f "$VIDEO_PATH" ]; then
        echo -e "${YELLOW}Skipping video test: bird.mp4 not found${NC}"
    else
        echo -e "${YELLOW}Skipping video test: ffmpeg not installed${NC}"
    fi
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoint: POST /v1/vision/classify"
echo ""
echo "Request format:"
echo "  {"
echo "    \"image\": \"<base64 encoded image>\","
echo "    \"labels\": [\"label1\", \"label2\", ...]"
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"scores\": {\"label1\": 0.85, \"label2\": 0.10, ...}"
echo "  }"
echo ""
