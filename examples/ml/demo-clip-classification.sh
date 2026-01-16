#!/bin/bash
# Demo: Zero-Shot Image Classification with CLIP
#
# This demo shows how to use CLIP for zero-shot image classification.
# No training required - just provide labels and classify images.
#
# Phase 5 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-clip-classification.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
echo -e "${BLUE}  Zero-Shot Image Classification Demo (CLIP)${NC}"
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

# Find test images
IMAGES_DIR="$SCRIPT_DIR/../files"
if [ ! -d "$IMAGES_DIR" ]; then
    echo -e "${RED}Error: Test images directory not found: ${IMAGES_DIR}${NC}"
    exit 1
fi

# Helper to classify an image using file path
classify_image() {
    local image_path="$1"
    local labels="$2"
    local image_name=$(basename "$image_path")

    # Use Python to handle the request - resize large images to avoid memory issues
    cd "$SCRIPT_DIR/../../runtimes/universal" && uv run python - <<EOF
import base64
import json
import urllib.request
from io import BytesIO
from PIL import Image

# Load and resize image if needed (max 512px on longest side for demo)
img = Image.open("$image_path").convert("RGB")
max_size = 512
if max(img.size) > max_size:
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

# Encode to base64
buffer = BytesIO()
img.save(buffer, format="JPEG", quality=85)
image_b64 = base64.b64encode(buffer.getvalue()).decode()

# Make request
data = json.dumps({
    "image": image_b64,
    "labels": $labels
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/classify-zero-shot",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

print(f"Predicted: {result['label']} (score: {result['score']:.3f})")
print("All scores:")
for label, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 20)
    print(f"  {label:10s}: {score:.3f} {bar}")
EOF
}

# Test 1: Classify cat image
echo -e "${BLUE}Test 1: Classify cat image${NC}"
echo -e "${YELLOW}Labels: cat, dog, horse, bird, fish, bear${NC}"
echo ""

CAT_IMAGE="$IMAGES_DIR/cat.png"
if [ ! -f "$CAT_IMAGE" ]; then
    CAT_IMAGE="$IMAGES_DIR/cat1.jpg"
fi

if [ ! -f "$CAT_IMAGE" ]; then
    echo -e "${RED}Error: No cat image found${NC}"
    exit 1
fi

classify_image "$CAT_IMAGE" '["cat", "dog", "horse", "bird", "fish", "bear"]'
echo ""

# Test 2: Classify horse image
echo -e "${BLUE}Test 2: Classify horse image${NC}"
echo -e "${YELLOW}Labels: cat, dog, horse, bird, fish, bear${NC}"
echo ""

HORSE_IMAGE="$IMAGES_DIR/horse.jpg"
if [ ! -f "$HORSE_IMAGE" ]; then
    echo -e "${YELLOW}Skipping: horse.jpg not found${NC}"
else
    classify_image "$HORSE_IMAGE" '["cat", "dog", "horse", "bird", "fish", "bear"]'
fi
echo ""

# Test 3: Custom document classification labels
echo -e "${BLUE}Test 3: Classify cat as document type (real-world use case)${NC}"
echo -e "${YELLOW}Labels: receipt, invoice, contract, id card, photo${NC}"
echo ""

classify_image "$CAT_IMAGE" '["receipt", "invoice", "contract", "id card", "photo"]'
echo ""

# Test 4: Batch classification
echo -e "${BLUE}Test 4: Batch classification (multiple images)${NC}"
echo -e "${YELLOW}Classifying multiple images at once${NC}"
echo ""

cd "$SCRIPT_DIR/../../runtimes/universal" && uv run python - <<EOF
import base64
import json
import urllib.request
import os
from io import BytesIO
from PIL import Image

# Find available images
images_dir = "$IMAGES_DIR"
image_files = []
for fname in ["cat.png", "cat1.jpg", "horse.jpg", "cat2.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        image_files.append(path)
        if len(image_files) >= 3:
            break

if len(image_files) < 2:
    print("Not enough images for batch test")
    exit(0)

# Encode images (resize to avoid memory issues)
def resize_and_encode(path):
    img = Image.open(path).convert("RGB")
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

images_b64 = [resize_and_encode(p) for p in image_files]

# Make batch request
data = json.dumps({
    "images": images_b64,
    "labels": ["cat", "dog", "horse", "bird"]
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/classify-zero-shot/batch",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())

print(f"Classified {result['total_count']} images:")
for i, (path, res) in enumerate(zip(image_files, result['data'])):
    fname = os.path.basename(path)
    print(f"  {fname}: {res['label']} (score: {res['score']:.3f})")
EOF
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "CLIP enables zero-shot classification - no training needed!"
echo "Just provide any labels and classify images instantly."
