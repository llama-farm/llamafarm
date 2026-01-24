#!/bin/bash
# Full Anomaly Detection Demo
# Runs all demo scripts in sequence with server management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

    # Kill by port
    lsof -ti:8005 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:11540 2>/dev/null | xargs kill -9 2>/dev/null || true

    # Kill nx processes
    pkill -9 -f "nx start universal-runtime" 2>/dev/null || true
    pkill -9 -f "nx start server" 2>/dev/null || true

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

# Main execution
print_header "LlamaFarm Anomaly Detection - Full Demo"

echo "This demo will:"
echo "  1. Start LlamaFarm servers"
echo "  2. Generate training data"
echo "  3. Train an anomaly detection model"
echo "  4. Run streaming detection with anomaly injection"
echo "  5. Demonstrate Polars buffer API"
echo ""

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Step 1: Stop any existing servers
print_header "Step 0: Stopping Existing Servers"
cleanup
sleep 2

# Step 2: Start servers
print_header "Step 1: Starting Servers"

cd "$REPO_ROOT"

print_step "Starting Universal Runtime..."
nx start universal-runtime > /tmp/universal-runtime.log 2>&1 &
RUNTIME_PID=$!

print_step "Waiting for Universal Runtime to initialize..."
sleep 8

print_step "Starting LlamaFarm Server..."
nx start server > /tmp/llamafarm-server.log 2>&1 &
SERVER_PID=$!

# Wait for servers to be ready
wait_for_server "http://localhost:11540/health" "Universal Runtime"
wait_for_server "http://localhost:8005/health" "LlamaFarm Server"

print_step "Both servers are running!"
echo ""

# Change to demo directory
cd "$SCRIPT_DIR"

# Step 3: Generate training data
print_header "Step 2: Generate Training Data"
python3 01_generate_training_data.py
echo ""

# Step 4: Train model
print_header "Step 3: Train Anomaly Detection Model"
cd "$REPO_ROOT/server" && uv run python "$SCRIPT_DIR/02_train_model.py"
cd "$SCRIPT_DIR"
echo ""

# Step 5: Streaming detection
print_header "Step 4: Streaming Anomaly Detection"
cd "$REPO_ROOT/server" && uv run python "$SCRIPT_DIR/03_streaming_detection.py"
cd "$SCRIPT_DIR"
echo ""

# Step 6: Polars features
print_header "Step 5: Polars Buffer API Demo"
cd "$REPO_ROOT/server" && uv run python "$SCRIPT_DIR/04_polars_features.py"
cd "$SCRIPT_DIR"
echo ""

# Summary
print_header "Demo Complete!"

echo -e "${GREEN}All demo scripts ran successfully!${NC}"
echo ""
echo "What was demonstrated:"
echo "  ✓ Training data generation (500 normal samples)"
echo "  ✓ Batch model training with ECOD backend"
echo "  ✓ Model saving, loading, and listing"
echo "  ✓ Streaming anomaly detection with cold start"
echo "  ✓ Automatic model retraining"
echo "  ✓ Anomaly injection and detection"
echo "  ✓ Polars buffer creation and management"
echo "  ✓ Rolling feature computation"
echo ""
echo "Server logs:"
echo "  Universal Runtime: /tmp/universal-runtime.log"
echo "  LlamaFarm Server:  /tmp/llamafarm-server.log"
echo ""
echo "To run individual scripts:"
echo "  python 01_generate_training_data.py"
echo "  python 02_train_model.py"
echo "  python 03_streaming_detection.py"
echo "  python 04_polars_features.py"
echo ""

# Cleanup will happen via trap
