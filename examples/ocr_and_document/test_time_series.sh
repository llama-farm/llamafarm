#!/bin/bash
# Test Time-Series Forecasting (Chronos) and Change Point Detection
#
# This script demonstrates:
# 1. Time-series forecasting
# 2. Change point detection
#
# Usage: ./test_time_series.sh [PORT]
#   PORT defaults to 11540 (Universal Runtime)

set -e

PORT=${1:-11540}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Time-Series Analysis Test${NC}"
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
# Test 1: Time-Series Forecasting
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Time-Series Forecasting${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Generate sample time series (sine wave with trend)
TIME_SERIES=$(python3 -c "
import math
import random
random.seed(42)
data = []
for i in range(50):
    val = 100 + 10 * math.sin(i * 0.3) + i * 0.5 + random.uniform(-2, 2)
    data.append(round(val, 2))
print(str(data).replace(' ', ''))
")

echo -e "${YELLOW}Input time series (50 points):${NC}"
echo "  Sine wave with upward trend + noise"
echo ""

echo -e "${YELLOW}Forecasting next 10 points...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/timeseries/forecast" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d "{
        \"data\": ${TIME_SERIES},
        \"horizon\": 10
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'forecast' in data:
        forecast = data['forecast']
        print(f'  ✓ Generated {len(forecast)} forecasted values:')
        for i, val in enumerate(forecast, 1):
            print(f'    t+{i}: {val:.2f}')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Change Point Detection
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Change Point Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Generate data with change points
CHANGE_DATA=$(python3 -c "
import random
random.seed(42)
# Segment 1: mean=10
seg1 = [10 + random.uniform(-1, 1) for _ in range(30)]
# Segment 2: mean=25
seg2 = [25 + random.uniform(-1, 1) for _ in range(30)]
# Segment 3: mean=15
seg3 = [15 + random.uniform(-1, 1) for _ in range(30)]
data = seg1 + seg2 + seg3
print(str([round(x, 2) for x in data]).replace(' ', ''))
")

echo -e "${YELLOW}Input data: 90 points with 2 change points${NC}"
echo "  Segment 1 (0-29): mean ~10"
echo "  Segment 2 (30-59): mean ~25"
echo "  Segment 3 (60-89): mean ~15"
echo ""

echo -e "${YELLOW}Detecting change points...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/timeseries/changepoints" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"data\": ${CHANGE_DATA},
        \"algorithm\": \"binseg\",
        \"n_changepoints\": 2
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'change_points' in data:
        cps = data['change_points']
        print(f'  ✓ Detected {len(cps)} change point(s):')
        for cp in cps:
            print(f'    - Position {cp} (expected ~30, ~60)')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 3: Different Change Point Algorithms
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Algorithm Comparison${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Comparing change point algorithms...${NC}"

for ALGO in binseg window bottomup; do
    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/timeseries/changepoints" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"data\": ${CHANGE_DATA},
            \"algorithm\": \"${ALGO}\",
            \"n_changepoints\": 2
        }")

    CPS=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    cps = data.get('change_points', [])
    print(', '.join(str(cp) for cp in cps))
except:
    print('error')
" 2>/dev/null)

    echo "  ${ALGO}: ${CPS}"
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
echo "  POST /v1/timeseries/forecast"
echo "    {\"data\": [...], \"horizon\": 10}"
echo ""
echo "  POST /v1/timeseries/changepoints"
echo "    {\"data\": [...], \"algorithm\": \"binseg\", \"n_changepoints\": 2}"
echo ""
echo "Algorithms: binseg, window, bottomup, pelt"
echo ""
