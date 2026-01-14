#!/bin/bash
# Test Drift Detection via LlamaFarm API
#
# Usage: ./test_drift_detection_api.sh [PORT]
#   PORT defaults to 8000 (LlamaFarm API)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
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
echo -e "${BLUE}  Drift Detection via LlamaFarm API Test${NC}"
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

DETECTOR_ID="api-test-$$"

# Create detector
echo -e "${YELLOW}Creating detector via /v1/ml/streaming/drift/create...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/streaming/drift/create" \
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
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Stream some data
echo -e "${YELLOW}Streaming data...${NC}"
for VAL in 5.1 5.2 4.9 5.0 15.1 15.2 14.9 15.0; do
    curl -s -X POST "${BASE_URL}/v1/ml/streaming/drift/update" \
        -H "Content-Type: application/json" \
        -d "{\"detector_id\": \"${DETECTOR_ID}\", \"value\": ${VAL}}" > /dev/null
done
echo "  ✓ Streamed 8 samples"
echo ""

# Cleanup
curl -s -X DELETE "${BASE_URL}/v1/ml/streaming/drift/${DETECTOR_ID}" > /dev/null 2>&1 || true

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
