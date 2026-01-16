#!/bin/bash
# Demo: Background Removal with RMBG
#
# This demo shows how to remove backgrounds from images using RMBG
# (Remove Background), a state-of-the-art background removal model.
#
# Phase 10 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-background-removal.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/../../runtimes/universal"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env" 2>/dev/null || true
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Background Removal Demo (RMBG)${NC}"
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

# Create a test image with a simple foreground object
echo -e "${YELLOW}Creating test image with foreground object...${NC}"
TEST_IMAGE=$(cd "$RUNTIME_DIR" && uv run python << 'EOF'
import base64
from io import BytesIO
from PIL import Image, ImageDraw

# Create an image with a colored background and a simple foreground
img = Image.new("RGB", (256, 256), color=(135, 206, 235))  # Sky blue background
draw = ImageDraw.Draw(img)

# Draw a red circle in the center (foreground)
draw.ellipse([78, 78, 178, 178], fill=(200, 50, 50))

# Draw a smaller yellow circle inside
draw.ellipse([98, 98, 158, 158], fill=(255, 220, 50))

# Resize to max 256px
max_size = 256
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

buffer = BytesIO()
img.save(buffer, format="PNG")
print(base64.b64encode(buffer.getvalue()).decode())
EOF
)
echo -e "${GREEN}✓ Test image created (256x256 with circles)${NC}"
echo ""

# Test 1: Basic Background Removal
echo -e "${BLUE}Test 1: Remove background from image${NC}"
echo -e "${YELLOW}Processing image...${NC}"
echo ""

RESULT=$(curl -s -X POST "${BASE_URL}/v1/vision/segment" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${TEST_IMAGE}\",
    \"return_mask\": false
  }")
echo "$RESULT" | (cd "$RUNTIME_DIR" && uv run python -c "
import json, sys, base64
from PIL import Image
from io import BytesIO

data = json.load(sys.stdin)
if 'error' in data or 'detail' in data:
    print(f\"Error: {data.get('detail', data.get('error', 'Unknown error'))}\")
    sys.exit(1)

print(f\"Output dimensions: {data['width']}x{data['height']}\")

# Decode and check the output image
img_data = base64.b64decode(data['image'])
img = Image.open(BytesIO(img_data))
print(f\"Output format: {img.format}\")
print(f\"Output mode: {img.mode} (RGBA = has alpha channel)\")

# Check alpha channel stats
if img.mode == 'RGBA':
    alpha = img.split()[3]
    alpha_min = min(alpha.getdata())
    alpha_max = max(alpha.getdata())
    print(f\"Alpha range: {alpha_min} to {alpha_max} (0=transparent, 255=opaque)\")
")
echo ""

# Test 2: Get mask separately
echo -e "${BLUE}Test 2: Remove background and get mask${NC}"
echo -e "${YELLOW}Processing with return_mask=true...${NC}"
echo ""

RESULT=$(curl -s -X POST "${BASE_URL}/v1/vision/segment" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${TEST_IMAGE}\",
    \"return_mask\": true
  }")
echo "$RESULT" | (cd "$RUNTIME_DIR" && uv run python -c "
import json, sys, base64
from PIL import Image
from io import BytesIO

data = json.load(sys.stdin)
if 'error' in data or 'detail' in data:
    print(f\"Error: {data.get('detail', data.get('error', 'Unknown error'))}\")
    sys.exit(1)

print(f\"Got image: {len(data['image'])} bytes (base64)\")
print(f\"Got mask: {'mask' in data}\")

if 'mask' in data:
    mask_data = base64.b64decode(data['mask'])
    mask = Image.open(BytesIO(mask_data))
    print(f\"Mask format: {mask.format}\")
    print(f\"Mask mode: {mask.mode} (L = grayscale)\")
    print(f\"Mask size: {mask.size[0]}x{mask.size[1]}\")
")
echo ""

# Test 3: Product-like image
echo -e "${BLUE}Test 3: Product image simulation${NC}"
echo -e "${YELLOW}Creating product-style image...${NC}"
echo ""

PRODUCT_IMAGE=$(cd "$RUNTIME_DIR" && uv run python << 'EOF'
import base64
from io import BytesIO
from PIL import Image, ImageDraw

# Create a "product" image - item on background
img = Image.new("RGB", (256, 256), color=(240, 240, 245))  # Light gray background
draw = ImageDraw.Draw(img)

# Draw a "bottle" shape
draw.rectangle([100, 50, 156, 200], fill=(50, 120, 200))  # Blue bottle body
draw.rectangle([110, 30, 146, 55], fill=(70, 140, 220))  # Bottle neck
draw.ellipse([115, 15, 141, 35], fill=(80, 80, 80))  # Cap

# Add some shadow
draw.ellipse([90, 195, 166, 215], fill=(200, 200, 200))

buffer = BytesIO()
img.save(buffer, format="PNG")
print(base64.b64encode(buffer.getvalue()).decode())
EOF
)

RESULT=$(curl -s -X POST "${BASE_URL}/v1/vision/segment" \
  -H "Content-Type: application/json" \
  -d "{
    \"image\": \"${PRODUCT_IMAGE}\",
    \"return_mask\": false
  }")
echo "$RESULT" | (cd "$RUNTIME_DIR" && uv run python -c "
import json, sys, base64
from PIL import Image
from io import BytesIO

data = json.load(sys.stdin)
if 'error' in data or 'detail' in data:
    print(f\"Error: {data.get('detail', data.get('error', 'Unknown error'))}\")
    sys.exit(1)

print(f\"Product image processed: {data['width']}x{data['height']}\")

img_data = base64.b64decode(data['image'])
img = Image.open(BytesIO(img_data))
print(f\"Output mode: {img.mode}\")

if img.mode == 'RGBA':
    alpha = img.split()[3]
    transparent_pixels = sum(1 for p in alpha.getdata() if p < 128)
    total_pixels = img.size[0] * img.size[1]
    print(f\"Transparent pixels: {transparent_pixels}/{total_pixels} ({100*transparent_pixels/total_pixels:.1f}%)\")
")
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Background Removal (RMBG) supports:"
echo "  - High-quality alpha mask generation"
echo "  - PNG output with transparent background"
echo "  - Optional separate mask output"
echo "  - Product photography, portraits, any image"
echo "  - Batch processing for multiple images"
