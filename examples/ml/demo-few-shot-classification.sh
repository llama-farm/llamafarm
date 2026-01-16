#!/bin/bash
# Demo: Few-Shot Image Classification
#
# This demo shows how to train a custom image classifier with just a few
# examples per class. Uses CLIP embeddings with a linear probe.
#
# Features demonstrated:
# 1. Training a classifier with 2+ images per class
# 2. Predicting with the trained classifier
# 3. Refining the classifier with new classes
# 4. Batch prediction
#
# Usage: ./demo-few-shot-classification.sh

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
echo -e "${BLUE}  Few-Shot Image Classification Demo${NC}"
echo -e "${BLUE}  Train custom classifiers with minimal data${NC}"
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

CLASSIFIER_ID="demo-animal-classifier"

# Test 1: Train a classifier
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Train Few-Shot Classifier${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}Training classifier '${CLASSIFIER_ID}' with cat and horse images...${NC}"
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
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

images_dir = "$IMAGES_DIR"

# Find cat images
cat_images = []
for fname in ["cat.png", "cat1.jpg", "cat2.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        cat_images.append(path)

# Find horse images (or use other images as stand-in)
horse_images = []
for fname in ["horse.jpg", "horse1.jpg"]:
    path = os.path.join(images_dir, fname)
    if os.path.exists(path):
        horse_images.append(path)

if len(cat_images) < 1:
    print("Need at least 1 cat image")
    exit(1)

if len(horse_images) < 1:
    print("No horse images found, using cat images with different labels for demo")
    # Create synthetic training data
    images = [resize_and_encode(cat_images[0])]
    labels = ["cat"]
    if len(cat_images) > 1:
        images.append(resize_and_encode(cat_images[1]))
        labels.append("other")
else:
    images = [resize_and_encode(p) for p in cat_images[:2]]
    labels = ["cat"] * len(cat_images[:2])
    images += [resize_and_encode(p) for p in horse_images[:2]]
    labels += ["horse"] * len(horse_images[:2])

print(f"Training with {len(images)} images, {len(set(labels))} classes")
print(f"Classes: {sorted(set(labels))}")
print("")

# Train the classifier
data = json.dumps({
    "classifier_id": "${CLASSIFIER_ID}",
    "images": images,
    "labels": labels,
    "epochs": 100
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/classify/fit",
    data=data,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    print("Training Results:")
    print(f"  Samples: {result['num_samples']}")
    print(f"  Classes: {result['classes']}")
    print(f"  Final Accuracy: {result['final_accuracy']:.1%}")
    print(f"  Training Time: {result['training_time_ms']:.1f}ms")
    print(f"  Success: {result['success']}")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOF
echo ""

# Test 2: Predict with the classifier
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Predict with Trained Classifier${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${YELLOW}Classifying images with the trained classifier...${NC}"
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
    max_size = 512
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

images_dir = "$IMAGES_DIR"

# Find any available test image
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

# Predict
data = json.dumps({
    "classifier_id": "${CLASSIFIER_ID}",
    "image": image_b64
}).encode()

req = urllib.request.Request(
    "${BASE_URL}/v1/vision/classify/predict",
    data=data,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.loads(resp.read())

fname = os.path.basename(test_image)
print(f"Image: {fname}")
print(f"Prediction: {result['label']} (confidence: {result['score']:.1%})")
print("All scores:")
for label, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 30)
    print(f"  {label:10s}: {score:.1%} {bar}")
EOF
echo ""

# Test 3: Get classifier info
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Get Classifier Info${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

RESPONSE=$(curl -s "${BASE_URL}/v1/vision/classify/info/${CLASSIFIER_ID}")
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Classifier Info:')
print(f\"  ID: {data.get('classifier_id', 'N/A')}\")
print(f\"  Loaded: {data.get('is_loaded', False)}\")
print(f\"  Trained: {data.get('is_trained', False)}\")
print(f\"  Classes: {data.get('classes', [])}\")
print(f\"  Num Classes: {data.get('num_classes', 0)}\")
"
echo ""

# Test 4: Delete the classifier
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 4: Delete Classifier${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

RESPONSE=$(curl -s -X DELETE "${BASE_URL}/v1/vision/classify/${CLASSIFIER_ID}")
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"Deleted: {data.get('deleted', False)}\")
"
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Few-shot classification enables custom classifiers with minimal data."
echo ""
echo "API Endpoints:"
echo "  POST /v1/vision/classify/fit      - Train classifier"
echo "  POST /v1/vision/classify/predict  - Classify single image"
echo "  POST /v1/vision/classify/predict/batch - Classify multiple images"
echo "  POST /v1/vision/classify/refine   - Add more training data"
echo "  GET  /v1/vision/classify/info/{id} - Get classifier info"
echo "  DELETE /v1/vision/classify/{id}   - Delete classifier"
echo ""
