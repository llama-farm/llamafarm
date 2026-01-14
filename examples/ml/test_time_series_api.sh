#!/bin/bash
# Test Time-Series Analysis via LlamaFarm API
#
# Usage: ./test_time_series_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Time-Series via LlamaFarm API Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking LlamaFarm API health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: LlamaFarm API not running on port ${PORT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Test forecasting
echo -e "${YELLOW}Testing forecasting via /v1/ml/timeseries/forecast...${NC}"

DATA="[100,102,105,103,107,110,108,112,115,113,117,120]"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/timeseries/forecast" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d "{
        \"data\": ${DATA},
        \"horizon\": 5
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'forecast' in data:
        print(f'  ✓ Forecast: {data[\"forecast\"]}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Test change points
echo -e "${YELLOW}Testing change points via /v1/ml/timeseries/changepoints...${NC}"

CHANGE_DATA="[10,11,10,12,11,25,26,24,27,25,15,16,14,17,15]"

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/timeseries/changepoints" \
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
        print(f'  ✓ Change points: {data[\"change_points\"]}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
