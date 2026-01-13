#!/bin/bash
# Demo: Non-Blocking Async Training
#
# This script demonstrates that ML training runs asynchronously,
# allowing health checks and other requests to be served during training.
#
# What this tests:
# 1. Start a large training job (anomaly detection)
# 2. Immediately start hitting the health endpoint
# 3. Verify health checks respond quickly (< 500ms) during training
# 4. Verify training completes successfully
#
# Usage: ./demo-async-training.sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (or 11545)

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env" 2>/dev/null || true
fi
RUNTIME_PORT=${1:-${LF_RUNTIME_PORT:-11545}}
BASE_URL="http://localhost:${RUNTIME_PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Non-Blocking Async Training Demo${NC}"
echo -e "${BLUE}  Universal Runtime - port ${RUNTIME_PORT}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking Universal Runtime health...${NC}"
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${RUNTIME_PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# Generate training data (large dataset for longer training)
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Generating Training Data${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Python to generate data
TRAINING_DATA=$(python3 -c "
import json
import random
random.seed(42)
# 1000 samples, 20 features - this will take a few seconds to train
data = [[random.gauss(0, 1) for _ in range(20)] for _ in range(1000)]
print(json.dumps(data))
")

echo "Generated 1000 samples with 20 features"
echo ""

# ============================================================================
# Test: Start training and hit health endpoint during training
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test: Health Checks During Training${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Start training in background
echo -e "${YELLOW}Starting anomaly detector training (background)...${NC}"
TRAINING_START=$(date +%s.%N)

# Start the training request in background
curl -sf -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"async-test-detector\",
        \"backend\": \"one_class_svm\",
        \"data\": ${TRAINING_DATA},
        \"contamination\": 0.1
    }" > /tmp/training_result.json 2>&1 &
TRAINING_PID=$!

echo "Training started (PID: $TRAINING_PID)"
echo ""

# Give training a moment to start
sleep 0.5

# Hit health endpoint multiple times during training
echo -e "${YELLOW}Hitting health endpoint during training...${NC}"
HEALTH_TIMES=()
HEALTH_SUCCESS=0

for i in {1..5}; do
    START=$(date +%s.%N)
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        END=$(date +%s.%N)
        ELAPSED=$(echo "$END - $START" | bc)
        HEALTH_TIMES+=("$ELAPSED")
        echo "  Health check $i: ${ELAPSED}s"
        ((HEALTH_SUCCESS++)) || true
    else
        echo "  Health check $i: FAILED"
    fi
    sleep 0.3
done

# Wait for training to complete
echo ""
echo -e "${YELLOW}Waiting for training to complete...${NC}"
wait $TRAINING_PID
TRAINING_END=$(date +%s.%N)
TRAINING_ELAPSED=$(echo "$TRAINING_END - $TRAINING_START" | bc)

# Check training result
if [ -f /tmp/training_result.json ]; then
    TRAINING_RESULT=$(cat /tmp/training_result.json)
    if echo "$TRAINING_RESULT" | python3 -c "import sys, json; data=json.load(sys.stdin); sys.exit(0 if 'status' in data else 1)" 2>/dev/null; then
        echo -e "${GREEN}✓ Training completed successfully in ${TRAINING_ELAPSED}s${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        echo "$TRAINING_RESULT"
        exit 1
    fi
else
    echo -e "${RED}✗ No training result file${NC}"
    exit 1
fi

# ============================================================================
# Results
# ============================================================================
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Results${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Calculate average health check time
if [ ${#HEALTH_TIMES[@]} -gt 0 ]; then
    TOTAL=0
    for t in "${HEALTH_TIMES[@]}"; do
        TOTAL=$(echo "$TOTAL + $t" | bc)
    done
    AVG=$(echo "scale=3; $TOTAL / ${#HEALTH_TIMES[@]}" | bc)

    echo "Health checks during training:"
    echo "  Successful: $HEALTH_SUCCESS / 5"
    echo "  Average response time: ${AVG}s"
    echo ""

    # Check if health checks were fast enough (< 0.5s)
    if (( $(echo "$AVG < 0.5" | bc -l) )); then
        echo -e "${GREEN}✓ PASSED: Health checks responded quickly during training${NC}"
        echo "  (Average ${AVG}s < 0.5s threshold)"
    else
        echo -e "${RED}✗ FAILED: Health checks were too slow during training${NC}"
        echo "  (Average ${AVG}s >= 0.5s threshold)"
        echo "  This indicates training is blocking the event loop!"
        exit 1
    fi
else
    echo -e "${RED}✗ FAILED: No health checks succeeded${NC}"
    exit 1
fi

# Cleanup
echo ""
echo -e "${YELLOW}Cleaning up test model...${NC}"
curl -sf -X DELETE "${BASE_URL}/v1/anomaly/models/async-test-detector" > /dev/null 2>&1 || true
echo -e "${GREEN}✓ Cleanup complete${NC}"

echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Demo Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "This demo verified that:"
echo "  1. Training runs asynchronously in a thread pool"
echo "  2. Health checks respond quickly during training"
echo "  3. The event loop is NOT blocked during CPU-bound training"
echo ""
