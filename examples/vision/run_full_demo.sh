#!/bin/bash
# Full Vision Pipeline Demo
# Runs all demo scripts in sequence with server management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
RUNTIME_PORT="${RUNTIME_PORT:-14345}"

# Load from .env if exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

export RUNTIME_URL="${RUNTIME_URL:-http://localhost:$RUNTIME_PORT}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

cleanup() {
    print_header "Cleanup"
    print_step "Stopping servers..."
    lsof -ti:$RUNTIME_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    pkill -9 -f "nx start universal-runtime" 2>/dev/null || true
    print_step "Servers stopped"
}

wait_for_server() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    print_step "Waiting for $name to be ready..."

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    print_error "$name failed to start after ${max_attempts}s"
    return 1
}

# Main
print_header "LlamaFarm Vision Pipeline - Full Demo"

echo "This demo will:"
echo "  1. Start the Universal Runtime"
echo "  2. Run detection, classification, segmentation, and OCR"
echo "  3. Start a streaming session with cascade"
echo "  4. Demonstrate federation and model management"
echo ""

trap cleanup EXIT

# Step 0: Stop existing
print_header "Step 0: Stop Existing Servers"
cleanup
sleep 2

# Step 1: Start runtime
print_header "Step 1: Start Universal Runtime"

cd "$REPO_ROOT"
print_step "Starting Universal Runtime..."
nx start universal-runtime > /tmp/universal-runtime.log 2>&1 &

wait_for_server "http://localhost:$RUNTIME_PORT/health" "Universal Runtime"

# Step 2: Detection & Classification
print_header "Step 2: Detection, Classification, Segmentation, OCR"
cd "$REPO_ROOT/runtimes/universal" && uv run python "$SCRIPT_DIR/01_detect_and_classify.py"
cd "$SCRIPT_DIR"
echo ""

# Step 3: Streaming Cascade
print_header "Step 3: Streaming Cascade Pipeline"
cd "$REPO_ROOT/runtimes/universal" && uv run python "$SCRIPT_DIR/02_streaming_cascade.py"
cd "$SCRIPT_DIR"
echo ""

# Step 4: Federation & Models
print_header "Step 4: Federation & Model Management"
cd "$REPO_ROOT/runtimes/universal" && uv run python "$SCRIPT_DIR/03_federation_and_models.py"
cd "$SCRIPT_DIR"
echo ""

# Summary
print_header "Demo Complete!"

echo -e "${GREEN}All vision demo scripts ran successfully!${NC}"
echo ""
echo "What was demonstrated:"
echo "  ✓ Object detection with YOLOv8"
echo "  ✓ Zero-shot classification with CLIP"
echo "  ✓ Instance segmentation"
echo "  ✓ OCR text extraction"
echo "  ✓ Streaming sessions with cascade escalation"
echo "  ✓ Replay buffer for automatic learning"
echo "  ✓ Human corrections and review queue"
echo "  ✓ Model save/load/versioning"
echo "  ✓ Federation peer management"
echo "  ✓ Model packaging and distribution"
echo ""
echo "Server logs: /tmp/universal-runtime.log"
echo ""
echo "To run individual scripts:"
echo "  cd runtimes/universal"
echo "  uv run python ../../examples/vision/01_detect_and_classify.py"
echo "  uv run python ../../examples/vision/02_streaming_cascade.py"
echo "  uv run python ../../examples/vision/03_federation_and_models.py"
echo ""
