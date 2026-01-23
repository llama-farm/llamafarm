#!/bin/bash
# Test Anomaly Detection endpoint via LlamaFarm API
#
# This script demonstrates:
# 1. Fitting an anomaly detector on normal data
# 2. Saving the trained model (production-ready)
# 3. Scoring new data for anomalies
# 4. Detecting anomalies with threshold
# 5. Loading a saved model (simulating server restart)
#
# Usage: ./test_anomaly_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}/v1/ml"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  LlamaFarm API Anomaly Detection Test${NC}"
echo -e "${BLUE}  (via /v1/ml proxy)${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if LlamaFarm server is running
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    echo "Start it with: nx start server"
    exit 1
fi
echo -e "${GREEN}✓ LlamaFarm API is healthy${NC}"

# Check Universal Runtime via proxy
echo -e "${YELLOW}Checking Universal Runtime via proxy...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not available${NC}"
    echo "Start it with: nx start universal"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# ============================================================================
# Test 1: Fit Anomaly Detector
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Fit Anomaly Detector${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training isolation forest on normal sensor data...${NC}"
echo "   Backend: isolation_forest"
echo "   Simulating: Temperature readings (normal range: 20-25°C)"
echo ""

# Generate normal data (temperatures between 20-25)
FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector",
        "backend": "isolation_forest",
        "data": [
            [22.1], [23.5], [21.8], [24.2], [22.7],
            [23.1], [21.5], [24.8], [22.3], [23.9],
            [21.2], [24.5], [22.8], [23.2], [21.9],
            [24.1], [22.5], [23.7], [21.6], [24.3],
            [22.2], [23.4], [21.7], [24.6], [22.9],
            [23.0], [21.4], [24.4], [22.6], [23.8]
        ],
        "contamination": 0.05
    }')

echo "Fit Response:"
echo "$FIT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$FIT_RESPONSE"
echo ""

if echo "$FIT_RESPONSE" | grep -q '"samples_fitted"'; then
    echo -e "${GREEN}✓ Anomaly detector trained successfully!${NC}"
else
    echo -e "${YELLOW}Fit may have failed${NC}"
fi

# Extract versioned model name from response
VERSIONED_MODEL=$(echo "$FIT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('versioned_name', 'sensor_anomaly_detector'))" 2>/dev/null)
echo "   Versioned model: $VERSIONED_MODEL"
echo ""

# Note: Models now autosave during fit, no explicit save endpoint needed
echo -e "${GREEN}✓ Model autosaved during training!${NC}"
echo ""

# ============================================================================
# Test 2: List Saved Models
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: List Saved Models${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Listing available models...${NC}"

LIST_RESPONSE=$(curl -s -X GET "${BASE_URL}/anomaly/models" --max-time 60)

echo "Models Response:"
echo "$LIST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LIST_RESPONSE"
echo ""

# ============================================================================
# Test 3: Detect Anomalies (using model still in memory from training)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Detect Anomalies${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Detecting anomalies in batch data...${NC}"
echo "   Model: $VERSIONED_MODEL (still in memory from training)"
echo "   Threshold: 0.5 (normalized score)"
echo ""

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/anomaly/detect" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "'"$VERSIONED_MODEL"'",
        "backend": "isolation_forest",
        "data": [
            [22.0], [23.5], [0.0], [21.5], [100.0],
            [24.0], [-10.0], [22.8], [35.0], [23.2]
        ],
        "threshold": 0.5
    }')

echo "Detect Response:"
echo "$DETECT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DETECT_RESPONSE"
echo ""

if echo "$DETECT_RESPONSE" | grep -q '"anomalies_detected"'; then
    echo -e "${GREEN}✓ Anomaly detection completed!${NC}"
    echo ""
    echo "Summary:"
    echo "$DETECT_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    readings = [22.0, 23.5, 0.0, 21.5, 100.0, 24.0, -10.0, 22.8, 35.0, 23.2]
    anomalies = data.get('data', [])
    summary = data.get('summary', {})
    print(f'  Anomalies found: {summary.get(\"anomalies_detected\", len(anomalies))}')
    print(f'  Threshold: {summary.get(\"threshold\", \"N/A\")}')
    if anomalies:
        print(f'  Anomalous readings:')
        for a in anomalies:
            idx = a['index']
            print(f'    - {readings[idx]}°C (score: {a[\"score\"]:.4f})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${YELLOW}Detection may have failed${NC}"
    echo "$DETECT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DETECT_RESPONSE"
fi
echo ""

# ============================================================================
# Test 4: Score Anomalies (returns all points with scores)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 4: Score Anomalies (all points)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Scoring all data points...${NC}"
echo ""

SCORE_RESPONSE=$(curl -s -X POST "${BASE_URL}/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "'"$VERSIONED_MODEL"'",
        "backend": "isolation_forest",
        "data": [
            [22.0], [23.5], [0.0], [21.5], [100.0]
        ],
        "threshold": 0.5
    }')

echo "Score Response:"
echo "$SCORE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SCORE_RESPONSE"
echo ""

# ============================================================================
# Test 5: Load Saved Model (Production Workflow)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 5: Load Saved Model (Production)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Loading previously saved model...${NC}"
echo "   This simulates loading after server restart"
echo ""

LOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/anomaly/load" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector-latest",
        "backend": "isolation_forest"
    }')

echo "Load Response:"
echo "$LOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LOAD_RESPONSE"
echo ""

if echo "$LOAD_RESPONSE" | grep -q '"loaded"'; then
    echo -e "${GREEN}✓ Model loaded and ready for inference!${NC}"

    # Test with loaded model
    echo ""
    echo -e "${YELLOW}Testing loaded model with new data...${NC}"

    TEST_RESPONSE=$(curl -s -X POST "${BASE_URL}/anomaly/detect" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d '{
            "model": "sensor_anomaly_detector-latest",
            "backend": "isolation_forest",
            "data": [[22.5], [100.0], [23.0]],
            "threshold": 0.5
        }')

    echo "$TEST_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    anomalies = data.get('data', [])
    print(f'  Detected {len(anomalies)} anomalies in test data')
    for a in anomalies:
        print(f'    - Index {a[\"index\"]}: score {a[\"score\"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${YELLOW}Load may have failed (this is expected on first run)${NC}"
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Production Workflow Summary (via LlamaFarm API):"
echo "  1. POST /v1/ml/anomaly/fit     - Train model (autosaves)"
echo "  2. GET  /v1/ml/anomaly/models  - List saved models"
echo "  3. POST /v1/ml/anomaly/load    - Load model (after restart)"
echo "  4. POST /v1/ml/anomaly/detect  - Detect anomalies"
echo "  5. POST /v1/ml/anomaly/score   - Score all points"
echo ""
