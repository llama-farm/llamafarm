#!/bin/bash
# Demo: Object Detection with YOLOS
#
# This demo shows how to detect objects in images using YOLOS
# (You Only Look at One Sequence), a Vision Transformer-based detector.
#
# Phase 9 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-object-detection.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/../../runtimes/universal"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env" 2>/dev/null || true
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Use uv run for Python commands to get access to PIL
PYTHON_CMD="cd $RUNTIME_DIR && uv run python"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Object Detection Demo (YOLOS)${NC}"
echo -e "${BLUE}  Port: ${PORT}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Health check
echo -e "${YELLOW}Checking Universal Runtime health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# Create a test image with some shapes (simulating objects)
echo -e "${YELLOW}Creating test image with shapes...${NC}"
TEST_IMAGE=$(cd "$RUNTIME_DIR" && uv run python << 'EOF'
import base64
from io import BytesIO
from PIL import Image, ImageDraw

# Create an image with some simple shapes
img = Image.new("RGB", (400, 300), color=(135, 206, 235))  # Sky blue background
draw = ImageDraw.Draw(img)

# Draw a "car" (rectangle)
draw.rectangle([50, 200, 150, 260], fill=(200, 0, 0), outline=(100, 0, 0))
draw.ellipse([60, 250, 90, 280], fill=(30, 30, 30))  # Wheel
draw.ellipse([110, 250, 140, 280], fill=(30, 30, 30))  # Wheel

# Draw a "person" (stick figure area)
draw.rectangle([200, 150, 240, 260], fill=(255, 200, 150))  # Body area
draw.ellipse([205, 120, 235, 150], fill=(255, 200, 150))  # Head

# Draw a "dog" shape
draw.ellipse([300, 200, 370, 250], fill=(139, 69, 19))  # Body
draw.ellipse([355, 190, 385, 220], fill=(139, 69, 19))  # Head

# Resize to max 512px to avoid transfer issues
max_size = 512
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)

# Convert to base64
buffer = BytesIO()
img.save(buffer, format="PNG")
b64 = base64.b64encode(buffer.getvalue()).decode()
print(b64)
EOF
)
echo -e "${GREEN}✓ Test image created (400x300 with shapes)${NC}"
echo ""

# Test 1: Basic Object Detection
echo -e "${BLUE}Test 1: Detect all objects in image${NC}"
echo -e "${YELLOW}Using default threshold (0.5)${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/vision/detect-objects" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${TEST_IMAGE}\",
    \"threshold\": 0.3
  }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Detected {data['count']} objects in {data['image_size']['width']}x{data['image_size']['height']} image:\")
if data['count'] == 0:
    print('  (No objects detected above threshold)')
for obj in data['objects'][:10]:  # Show top 10
    box = obj['box']
    print(f\"  - {obj['label']}: {obj['score']:.2f} @ [{box['x1']:.0f},{box['y1']:.0f},{box['x2']:.0f},{box['y2']:.0f}]\")
"
echo ""

# Test 2: Low Threshold Detection
echo -e "${BLUE}Test 2: Detect with low threshold (0.1)${NC}"
echo -e "${YELLOW}Lower threshold = more detections${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/vision/detect-objects" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${TEST_IMAGE}\",
    \"threshold\": 0.1
  }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Detected {data['count']} objects with threshold=0.1:\")
for obj in data['objects'][:15]:  # Show top 15
    print(f\"  - {obj['label']}: {obj['score']:.3f}\")
"
echo ""

# Test 3: Filter by Labels
echo -e "${BLUE}Test 3: Filter to specific object types${NC}"
echo -e "${YELLOW}Only looking for: person, car, dog, cat${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/vision/detect-objects" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${TEST_IMAGE}\",
    \"threshold\": 0.1,
    \"labels\": [\"person\", \"car\", \"dog\", \"cat\"]
  }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Found {data['count']} objects matching filter:\")
for obj in data['objects']:
    box = obj['box']
    print(f\"  - {obj['label']}: {obj['score']:.3f} @ [{box['x1']:.0f},{box['y1']:.0f},{box['x2']:.0f},{box['y2']:.0f}]\")
if data['count'] == 0:
    print('  (No matching objects found)')
"
echo ""

# Test 4: Create a more realistic scene
echo -e "${BLUE}Test 4: Street scene with multiple objects${NC}"
echo -e "${YELLOW}Creating synthetic street scene...${NC}"
echo ""

STREET_IMAGE=$(cd "$RUNTIME_DIR" && uv run python << 'EOF'
import base64
from io import BytesIO
from PIL import Image, ImageDraw

# Create a street scene
img = Image.new("RGB", (640, 480), color=(100, 150, 200))  # Sky
draw = ImageDraw.Draw(img)

# Ground/road
draw.rectangle([0, 300, 640, 480], fill=(80, 80, 80))  # Road
draw.rectangle([0, 280, 640, 300], fill=(34, 139, 34))  # Grass strip

# Buildings (background)
draw.rectangle([20, 150, 120, 280], fill=(180, 180, 180))  # Building 1
draw.rectangle([140, 100, 260, 280], fill=(200, 180, 160))  # Building 2
draw.rectangle([500, 120, 620, 280], fill=(170, 170, 190))  # Building 3

# Cars
draw.rectangle([50, 350, 150, 400], fill=(200, 50, 50))  # Red car
draw.ellipse([60, 390, 90, 420], fill=(30, 30, 30))
draw.ellipse([110, 390, 140, 420], fill=(30, 30, 30))

draw.rectangle([300, 340, 420, 400], fill=(50, 50, 200))  # Blue car
draw.ellipse([310, 390, 350, 420], fill=(30, 30, 30))
draw.ellipse([370, 390, 410, 420], fill=(30, 30, 30))

# People (simple stick figure representations)
# Person 1
draw.ellipse([200, 300, 220, 320], fill=(255, 220, 180))  # Head
draw.rectangle([205, 320, 215, 370], fill=(100, 100, 200))  # Body

# Person 2
draw.ellipse([450, 310, 470, 330], fill=(255, 220, 180))  # Head
draw.rectangle([455, 330, 465, 380], fill=(200, 100, 100))  # Body

# Resize to max 512px
max_size = 512
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)

buffer = BytesIO()
img.save(buffer, format="PNG")
print(base64.b64encode(buffer.getvalue()).decode())
EOF
)

curl -s -X POST "${BASE_URL}/v1/vision/detect-objects" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${STREET_IMAGE}\",
    \"threshold\": 0.2
  }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Street scene ({data['image_size']['width']}x{data['image_size']['height']}):\")
print(f\"Detected {data['count']} objects:\")
for obj in data['objects'][:10]:
    box = obj['box']
    print(f\"  - {obj['label']}: {obj['score']:.2f} @ [{box['x1']:.0f},{box['y1']:.0f},{box['x2']:.0f},{box['y2']:.0f}]\")
if data['count'] == 0:
    print('  (No objects detected - synthetic images may not match trained patterns)')
"
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Object Detection (YOLOS) supports:"
echo "  - 80 COCO object classes (person, car, dog, cat, bicycle, etc.)"
echo "  - Configurable confidence threshold"
echo "  - Label filtering for specific object types"
echo "  - Bounding box coordinates for each detection"
echo "  - Batch processing for multiple images"
