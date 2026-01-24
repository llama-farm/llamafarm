#!/usr/bin/env bash
# Demo: ADTK Time-Series Anomaly Detection
#
# This demo showcases:
# 1. Level shift detection (sudden baseline changes)
# 2. Spike detection (short-term outliers)
# 3. Volatility shift detection (variance changes)
# 4. Seasonal anomaly detection (pattern violations)
#
# Prerequisites:
# - Universal Runtime running on port 11540

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         LlamaFarm ADTK Time-Series Anomaly Detection         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration - Use LlamaFarm API (port 8005) for production usage
# Falls back to Universal Runtime (port 11545) for local testing
RUNTIME_URL="${LLAMAFARM_URL:-${RUNTIME_URL:-http://127.0.0.1:8005}}"

# Helper function
check_runtime() {
    echo -e "${YELLOW}Checking Universal Runtime availability...${NC}"
    if curl -s "${RUNTIME_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Universal Runtime is running at ${RUNTIME_URL}${NC}"
        return 0
    else
        echo -e "${RED}✗ Universal Runtime not available at ${RUNTIME_URL}${NC}"
        echo -e "${YELLOW}Start it with: nx start universal-runtime${NC}"
        return 1
    fi
}

# Check runtime
if ! check_runtime; then
    exit 1
fi

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: List available detectors${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "GET ${RUNTIME_URL}/v1/adtk/detectors"
curl -s "${RUNTIME_URL}/v1/adtk/detectors" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Detect LEVEL SHIFT (sudden baseline change)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate data with a level shift at day 15 (baseline goes from 100 to 150)
LEVEL_SHIFT_DATA='[
  {"timestamp": "2024-01-01", "value": 100},
  {"timestamp": "2024-01-02", "value": 101},
  {"timestamp": "2024-01-03", "value": 99},
  {"timestamp": "2024-01-04", "value": 102},
  {"timestamp": "2024-01-05", "value": 100},
  {"timestamp": "2024-01-06", "value": 101},
  {"timestamp": "2024-01-07", "value": 99},
  {"timestamp": "2024-01-08", "value": 100},
  {"timestamp": "2024-01-09", "value": 102},
  {"timestamp": "2024-01-10", "value": 99},
  {"timestamp": "2024-01-11", "value": 101},
  {"timestamp": "2024-01-12", "value": 100},
  {"timestamp": "2024-01-13", "value": 99},
  {"timestamp": "2024-01-14", "value": 101},
  {"timestamp": "2024-01-15", "value": 150},
  {"timestamp": "2024-01-16", "value": 152},
  {"timestamp": "2024-01-17", "value": 149},
  {"timestamp": "2024-01-18", "value": 151},
  {"timestamp": "2024-01-19", "value": 150},
  {"timestamp": "2024-01-20", "value": 148},
  {"timestamp": "2024-01-21", "value": 151},
  {"timestamp": "2024-01-22", "value": 150},
  {"timestamp": "2024-01-23", "value": 152},
  {"timestamp": "2024-01-24", "value": 149},
  {"timestamp": "2024-01-25", "value": 151}
]'

echo "POST ${RUNTIME_URL}/v1/adtk/detect"
echo "Data: Values around 100 for days 1-14, then jump to 150 on day 15"
curl -s -X POST "${RUNTIME_URL}/v1/adtk/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"detector\": \"level_shift\",
        \"data\": ${LEVEL_SHIFT_DATA},
        \"params\": {\"c\": 3.0, \"window\": 3}
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Detect SPIKE (sudden short-term outlier)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate data with a spike at day 10
SPIKE_DATA='[
  {"timestamp": "2024-01-01", "value": 100},
  {"timestamp": "2024-01-02", "value": 102},
  {"timestamp": "2024-01-03", "value": 99},
  {"timestamp": "2024-01-04", "value": 101},
  {"timestamp": "2024-01-05", "value": 100},
  {"timestamp": "2024-01-06", "value": 98},
  {"timestamp": "2024-01-07", "value": 102},
  {"timestamp": "2024-01-08", "value": 100},
  {"timestamp": "2024-01-09", "value": 101},
  {"timestamp": "2024-01-10", "value": 500},
  {"timestamp": "2024-01-11", "value": 99},
  {"timestamp": "2024-01-12", "value": 100},
  {"timestamp": "2024-01-13", "value": 102},
  {"timestamp": "2024-01-14", "value": 98},
  {"timestamp": "2024-01-15", "value": 101},
  {"timestamp": "2024-01-16", "value": 100},
  {"timestamp": "2024-01-17", "value": 99},
  {"timestamp": "2024-01-18", "value": 102},
  {"timestamp": "2024-01-19", "value": 100},
  {"timestamp": "2024-01-20", "value": 101}
]'

echo "POST ${RUNTIME_URL}/v1/adtk/detect"
echo "Data: Normal values around 100, with a spike to 500 on day 10"
curl -s -X POST "${RUNTIME_URL}/v1/adtk/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"detector\": \"spike\",
        \"data\": ${SPIKE_DATA},
        \"params\": {\"c\": 1.5}
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Detect VOLATILITY SHIFT (variance change)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate data with volatility shift at day 15 (low variance -> high variance)
VOLATILITY_DATA='[
  {"timestamp": "2024-01-01", "value": 100},
  {"timestamp": "2024-01-02", "value": 101},
  {"timestamp": "2024-01-03", "value": 100},
  {"timestamp": "2024-01-04", "value": 101},
  {"timestamp": "2024-01-05", "value": 100},
  {"timestamp": "2024-01-06", "value": 100},
  {"timestamp": "2024-01-07", "value": 101},
  {"timestamp": "2024-01-08", "value": 100},
  {"timestamp": "2024-01-09", "value": 100},
  {"timestamp": "2024-01-10", "value": 101},
  {"timestamp": "2024-01-11", "value": 100},
  {"timestamp": "2024-01-12", "value": 101},
  {"timestamp": "2024-01-13", "value": 100},
  {"timestamp": "2024-01-14", "value": 100},
  {"timestamp": "2024-01-15", "value": 90},
  {"timestamp": "2024-01-16", "value": 110},
  {"timestamp": "2024-01-17", "value": 85},
  {"timestamp": "2024-01-18", "value": 115},
  {"timestamp": "2024-01-19", "value": 80},
  {"timestamp": "2024-01-20", "value": 120},
  {"timestamp": "2024-01-21", "value": 88},
  {"timestamp": "2024-01-22", "value": 112},
  {"timestamp": "2024-01-23", "value": 82},
  {"timestamp": "2024-01-24", "value": 118},
  {"timestamp": "2024-01-25", "value": 86}
]'

echo "POST ${RUNTIME_URL}/v1/adtk/detect"
echo "Data: Low variance (100±1) for days 1-14, high variance (100±20) after day 15"
curl -s -X POST "${RUNTIME_URL}/v1/adtk/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"detector\": \"volatility_shift\",
        \"data\": ${VOLATILITY_DATA},
        \"params\": {\"c\": 3.0, \"window\": 5}
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Fit and save a detector model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "POST ${RUNTIME_URL}/v1/adtk/fit"
echo "Training a spike detector on reference data..."
curl -s -X POST "${RUNTIME_URL}/v1/adtk/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-spike-detector\",
        \"detector\": \"spike\",
        \"data\": ${SPIKE_DATA},
        \"description\": \"Demo spike detector for monitoring\"
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6: List saved models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "GET ${RUNTIME_URL}/v1/adtk/models"
curl -s "${RUNTIME_URL}/v1/adtk/models" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 7: Cleanup - Delete demo model${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "DELETE ${RUNTIME_URL}/v1/adtk/models/demo-spike-detector"
curl -s -X DELETE "${RUNTIME_URL}/v1/adtk/models/demo-spike-detector" | python3 -m json.tool
echo

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Demo Complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${YELLOW}Summary:${NC}"
echo "• Listed available ADTK detectors (level_shift, seasonal, spike, volatility_shift, persist, threshold)"
echo "• Detected level shift anomaly (baseline jump from 100 to 150)"
echo "• Detected spike anomaly (value jumped to 500)"
echo "• Detected volatility shift (variance increase)"
echo "• Trained and saved a spike detector model"
echo "• Cleaned up demo models"
echo
echo -e "${BLUE}ADTK detects temporal anomalies that point detectors miss:${NC}"
echo "• Level shifts: Sensor recalibration, system state changes"
echo "• Spikes: Short-term outliers, transient failures"
echo "• Volatility shifts: Increased noise, system instability"
echo "• Seasonal anomalies: Pattern violations in periodic data"
