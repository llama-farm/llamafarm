#!/bin/bash
# Demo: Open-Vocabulary Object Detection (OWL-ViT)
#
# This demo shows how to detect objects using natural language queries.
# No training required - describe what you want to find and detect it.
#
# Features demonstrated:
# 1. Text-conditioned detection (find objects by description)
# 2. Multiple queries at once
# 3. Threshold tuning
# 4. Image-guided detection (find similar objects)
#
# Usage: ./demo-open-vocab-detection.sh

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
echo -e "${BLUE}  Open-Vocabulary Object Detection Demo${NC}"
echo -e "${BLUE}  Find objects using natural language queries${NC}"
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

# Test 1: Basic text-conditioned detection
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Text-Conditioned Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}Finding objects by natural language description...${NC}"
echo ""

cd "$SCRIPT_DIR/../../runtimes/universal" && uv run python - <<EOF
import base64
import json
import urllib.request
import os
from io import BytesIO
from PIL import Image

def resize_and_encode(path):
    """Resize image and encode to base64."""
    img = Image.open(path).convert("RGB")
    max_size = 800  # OWL-ViT works better with larger images
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

images_dir = "$IMAGES_DIR"

# Find a test image
test_image = None
for fname in ["cat.png", "cat1.jpg", "horse.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        test_image = path
        break

if not test_image:
    print("No test images found")
    exit(1)

image_b64 = resize_and_encode(test_image)
fname = os.path.basename(test_image)

# Define queries - use descriptive phrases for better results
queries = [
    "a cat",
    "a dog",
    "a horse",
    "an animal"
]

print(f"Image: {fname}")
print(f"Queries: {queries}")
print("")

# Make detection request
data = json.dumps({
    "image": image_b64,
    "queries": queries,
    "threshold": 0.1,
    "top_k": 5
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/detect-open",
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    print(f"Found {result['count']} detections:")
    print("")

    if result['count'] == 0:
        print("  (No objects detected above threshold)")
    else:
        for obj in result['objects'][:5]:  # Show top 5
            box = obj['box']
            print(f"  {obj['query']}")
            print(f"    Score: {obj['score']:.1%}")
            print(f"    Box: ({box['x1']:.0f}, {box['y1']:.0f}) - ({box['x2']:.0f}, {box['y2']:.0f})")
            print("")

except Exception as e:
    print(f"Error: {e}")
EOF
echo ""

# Test 2: Species-specific detection
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Species-Specific Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}Using specific breed/species queries...${NC}"
echo ""

cd "$SCRIPT_DIR/../../runtimes/universal" && uv run python - <<EOF
import base64
import json
import urllib.request
import os
from io import BytesIO
from PIL import Image

def resize_and_encode(path):
    img = Image.open(path).convert("RGB")
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

images_dir = "$IMAGES_DIR"

# Find cat image for species test
test_image = None
for fname in ["cat.png", "cat1.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        test_image = path
        break

if not test_image:
    print("No cat images found")
    exit(0)

image_b64 = resize_and_encode(test_image)

# Specific breed queries
queries = [
    "a tabby cat",
    "a siamese cat",
    "an orange cat",
    "a black cat",
    "a white cat"
]

print(f"Image: {os.path.basename(test_image)}")
print(f"Testing breed detection with: {queries}")
print("")

data = json.dumps({
    "image": image_b64,
    "queries": queries,
    "threshold": 0.05,  # Lower threshold for specific breeds
    "top_k": 3
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/detect-open",
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    if result['count'] > 0:
        print("Best matches:")
        for obj in result['objects']:
            print(f"  {obj['query']}: {obj['score']:.1%}")
    else:
        print("No species-specific detections (try lower threshold)")

except Exception as e:
    print(f"Error: {e}")
EOF
echo ""

# Test 3: Detection with different thresholds
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Threshold Comparison${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}Comparing detection with different thresholds...${NC}"
echo ""

cd "$SCRIPT_DIR/../../runtimes/universal" && uv run python - <<EOF
import base64
import json
import urllib.request
import os
from io import BytesIO
from PIL import Image

def resize_and_encode(path):
    img = Image.open(path).convert("RGB")
    max_size = 800
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()

images_dir = "$IMAGES_DIR"

test_image = None
for fname in ["cat.png", "cat1.jpg", "horse.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        test_image = path
        break

if not test_image:
    print("No test images found")
    exit(0)

image_b64 = resize_and_encode(test_image)
queries = ["an animal", "a pet"]

print(f"Image: {os.path.basename(test_image)}")
print(f"Queries: {queries}")
print("")

for threshold in [0.3, 0.1, 0.05]:
    data = json.dumps({
        "image": image_b64,
        "queries": queries,
        "threshold": threshold
    }).encode()

    req = urllib.request.Request(
        "${BASE_URL}/v1/vision/detect-open",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    print(f"Threshold {threshold:.0%}: {result['count']} detections")
    if result['count'] > 0:
        top = result['objects'][0]
        print(f"  Best: {top['query']} ({top['score']:.1%})")
    print("")
EOF
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Open-vocabulary detection enables finding any object by description."
echo ""
echo "API Endpoints:"
echo "  POST /v1/vision/detect-open          - Text-conditioned detection"
echo "  POST /v1/vision/detect-open/batch    - Batch text detection"
echo "  POST /v1/vision/detect-open/by-image - Image-guided detection"
echo ""
echo "Tips:"
echo "  - Use descriptive queries: 'a photo of a cat' > 'cat'"
echo "  - Lower threshold (0.05-0.1) for recall, higher (0.3+) for precision"
echo "  - Combine with few-shot classification for species refinement"
echo ""
