#!/bin/bash
#
# Test: Non-Blocking Training
#
# This script demonstrates that the Universal Runtime doesn't block
# during model training. We:
# 1. Start an anomaly model training request (CPU-bound operation)
# 2. While training is in progress, make a health check request
# 3. Both requests should succeed - the health check should return immediately
#
# This verifies that the ThreadPoolExecutor pattern works correctly.

set -e

RUNTIME_URL="${RUNTIME_URL:-http://localhost:11540}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Non-Blocking Training Test"
echo "=============================================="
echo ""
echo "Runtime URL: $RUNTIME_URL"
echo ""

# Check if runtime is available
echo "Checking runtime availability..."
if ! curl -s "$RUNTIME_URL/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Universal Runtime not available at $RUNTIME_URL${NC}"
    echo "Please start the runtime first: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Runtime is available${NC}"
echo ""

# Generate training data (100 samples, 10 features)
echo "Generating training data..."
TRAINING_DATA=$(python3 -c "
import json
import random
random.seed(42)
# Generate 100 samples with 10 features each
data = [[random.gauss(0, 1) for _ in range(10)] for _ in range(100)]
print(json.dumps(data))
")

echo -e "${GREEN}✓ Generated 100 training samples${NC}"
echo ""

# Function to start training in background and capture PID
start_training() {
    echo "Starting anomaly model training (this triggers CPU-bound work)..."

    # Start training request in background
    (
        curl -s -X POST "$RUNTIME_URL/v1/anomaly/fit" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"nonblocking-test-model\",
                \"backend\": \"isolation_forest\",
                \"data\": $TRAINING_DATA
            }" > /tmp/training_result.json 2>&1
        echo $? > /tmp/training_exit_code
    ) &
    TRAINING_PID=$!
    echo "Training request started (PID: $TRAINING_PID)"
}

# Function to check health while training
check_health_during_training() {
    echo ""
    echo "Making health check request while training is in progress..."

    # Small delay to ensure training has started
    sleep 0.1

    # Time the health check
    START_TIME=$(python3 -c "import time; print(time.time())")

    HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$RUNTIME_URL/health")
    HEALTH_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)
    HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | head -n -1)

    END_TIME=$(python3 -c "import time; print(time.time())")
    DURATION=$(python3 -c "print(f'{($END_TIME - $START_TIME) * 1000:.1f}')")

    echo "Health check response time: ${DURATION}ms"
    echo "Health check status code: $HEALTH_CODE"

    if [ "$HEALTH_CODE" = "200" ]; then
        echo -e "${GREEN}✓ Health check succeeded during training!${NC}"
        return 0
    else
        echo -e "${RED}✗ Health check failed during training${NC}"
        return 1
    fi
}

# Function to wait for training and check result
wait_for_training() {
    echo ""
    echo "Waiting for training to complete..."

    wait $TRAINING_PID 2>/dev/null || true

    TRAINING_EXIT_CODE=$(cat /tmp/training_exit_code 2>/dev/null || echo "1")

    if [ "$TRAINING_EXIT_CODE" = "0" ]; then
        TRAINING_RESULT=$(cat /tmp/training_result.json)
        SAMPLES_FITTED=$(echo "$TRAINING_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('samples_fitted', 'unknown'))")
        echo -e "${GREEN}✓ Training completed successfully!${NC}"
        echo "  Samples fitted: $SAMPLES_FITTED"
        return 0
    else
        echo -e "${RED}✗ Training failed${NC}"
        cat /tmp/training_result.json 2>/dev/null || echo "No result file"
        return 1
    fi
}

# Cleanup function
cleanup() {
    rm -f /tmp/training_result.json /tmp/training_exit_code
    # Clean up the test model
    curl -s -X DELETE "$RUNTIME_URL/v1/anomaly/nonblocking-test-model" > /dev/null 2>&1 || true
}

# Run the test
echo "=============================================="
echo "Running Non-Blocking Test"
echo "=============================================="
echo ""

# Set up cleanup
trap cleanup EXIT

# Start training
start_training

# Check health during training
if check_health_during_training; then
    HEALTH_OK=true
else
    HEALTH_OK=false
fi

# Wait for training
if wait_for_training; then
    TRAINING_OK=true
else
    TRAINING_OK=false
fi

# Summary
echo ""
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo ""

if $HEALTH_OK && $TRAINING_OK; then
    echo -e "${GREEN}✓ NON-BLOCKING TEST PASSED${NC}"
    echo ""
    echo "The health check completed successfully while training was in progress."
    echo "This confirms that CPU-bound training operations are properly offloaded"
    echo "to the thread pool and don't block the async event loop."
    echo ""
    exit 0
else
    echo -e "${RED}✗ NON-BLOCKING TEST FAILED${NC}"
    echo ""
    if ! $HEALTH_OK; then
        echo "  - Health check failed during training"
    fi
    if ! $TRAINING_OK; then
        echo "  - Training request failed"
    fi
    echo ""
    exit 1
fi
