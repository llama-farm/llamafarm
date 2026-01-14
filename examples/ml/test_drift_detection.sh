#!/bin/bash
# Test Streaming Drift Detection (River)
#
# This script demonstrates:
# 1. Creating drift detectors
# 2. Streaming data and detecting drift
# 3. Different drift detection algorithms
#
# Usage: ./test_drift_detection.sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (fallback: 11540)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi


PORT=${1:-${LF_RUNTIME_PORT:-11540}}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Streaming Drift Detection Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# ============================================================================
# Test 1: ADWIN Drift Detection
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: ADWIN Drift Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

DETECTOR_ID="test-adwin-$$"

echo -e "${YELLOW}Creating ADWIN detector...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/streaming/drift/create" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"detector_id\": \"${DETECTOR_ID}\",
        \"algorithm\": \"adwin\"
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('created'):
        print('  ✓ Detector created')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${YELLOW}Streaming normal data (50 samples, mean=5)...${NC}"
NORMAL_DRIFTS=0
for i in $(seq 1 50); do
    VAL=$(python3 -c "import random; random.seed($i); print(5 + random.uniform(-0.5, 0.5))")
    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/streaming/drift/update" \
        -H "Content-Type: application/json" \
        --max-time 10 \
        -d "{\"detector_id\": \"${DETECTOR_ID}\", \"value\": ${VAL}}")

    DETECTED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_detected', False))" 2>/dev/null)
    if [ "$DETECTED" = "True" ]; then
        NORMAL_DRIFTS=$((NORMAL_DRIFTS + 1))
    fi
done
echo "  Drifts detected during normal data: ${NORMAL_DRIFTS}"

echo -e "${YELLOW}Streaming shifted data (50 samples, mean=15)...${NC}"
SHIFTED_DRIFTS=0
FIRST_DRIFT=""
for i in $(seq 51 100); do
    VAL=$(python3 -c "import random; random.seed($i); print(15 + random.uniform(-0.5, 0.5))")
    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/streaming/drift/update" \
        -H "Content-Type: application/json" \
        --max-time 10 \
        -d "{\"detector_id\": \"${DETECTOR_ID}\", \"value\": ${VAL}}")

    DETECTED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_detected', False))" 2>/dev/null)
    if [ "$DETECTED" = "True" ]; then
        SHIFTED_DRIFTS=$((SHIFTED_DRIFTS + 1))
        if [ -z "$FIRST_DRIFT" ]; then
            FIRST_DRIFT=$i
        fi
    fi
done
echo "  Drifts detected during shifted data: ${SHIFTED_DRIFTS}"
if [ -n "$FIRST_DRIFT" ]; then
    echo -e "  ${GREEN}✓ First drift detected at sample ${FIRST_DRIFT}${NC}"
fi
echo ""

# Get final stats
echo -e "${YELLOW}Detector statistics:${NC}"
RESPONSE=$(curl -s "${BASE_URL}/v1/streaming/drift/${DETECTOR_ID}/stats")
echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  Samples processed: {data.get(\"n_samples\", 0)}')
    print(f'  Total drifts: {data.get(\"n_drifts\", 0)}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Cleanup
curl -s -X DELETE "${BASE_URL}/v1/streaming/drift/${DETECTOR_ID}" > /dev/null 2>&1 || true

# ============================================================================
# Test 2: Compare Algorithms
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Compare Drift Algorithms${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Testing different algorithms on same data...${NC}"
echo ""

for ALGO in adwin page_hinkley kswin; do
    DET_ID="test-${ALGO}-$$"

    # Create detector
    curl -s -X POST "${BASE_URL}/v1/streaming/drift/create" \
        -H "Content-Type: application/json" \
        -d "{\"detector_id\": \"${DET_ID}\", \"algorithm\": \"${ALGO}\"}" > /dev/null

    # Stream data
    DRIFTS=0
    for i in $(seq 1 30); do
        VAL=$(python3 -c "import random; random.seed($i); print(5 + random.uniform(-0.5, 0.5))")
        RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/streaming/drift/update" \
            -H "Content-Type: application/json" \
            -d "{\"detector_id\": \"${DET_ID}\", \"value\": ${VAL}}")
    done

    for i in $(seq 31 60); do
        VAL=$(python3 -c "import random; random.seed($i); print(15 + random.uniform(-0.5, 0.5))")
        RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/streaming/drift/update" \
            -H "Content-Type: application/json" \
            -d "{\"detector_id\": \"${DET_ID}\", \"value\": ${VAL}}")

        DETECTED=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_detected', False))" 2>/dev/null)
        if [ "$DETECTED" = "True" ]; then
            DRIFTS=$((DRIFTS + 1))
        fi
    done

    echo "  ${ALGO}: ${DRIFTS} drift(s) detected"

    # Cleanup
    curl -s -X DELETE "${BASE_URL}/v1/streaming/drift/${DET_ID}" > /dev/null 2>&1 || true
done
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoints:"
echo ""
echo "  POST /v1/streaming/drift/create"
echo "    {\"detector_id\": \"my-detector\", \"algorithm\": \"adwin\"}"
echo ""
echo "  POST /v1/streaming/drift/update"
echo "    {\"detector_id\": \"my-detector\", \"value\": 5.0}"
echo ""
echo "  GET /v1/streaming/drift/{detector_id}/stats"
echo ""
echo "  DELETE /v1/streaming/drift/{detector_id}"
echo ""
echo "Algorithms:"
echo "  adwin        - Adaptive Windowing (most popular)"
echo "  page_hinkley - Page-Hinkley test for mean shifts"
echo "  kswin        - Kolmogorov-Smirnov windowing"
echo "  ddm          - Drift Detection Method (for classifiers)"
echo ""
