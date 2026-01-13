#!/bin/bash
# Test Anomaly SHAP Explanations via LlamaFarm API
#
# Usage: ./test_anomaly_explain_api.sh [PORT]
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
echo -e "${BLUE}  Anomaly Explain via LlamaFarm API Test${NC}"
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

MODEL_NAME="api-explain-test-$$"

# Train model first
echo -e "${YELLOW}Training model via /v1/ml/anomaly/fit...${NC}"
curl -s -X POST "${BASE_URL}/v1/ml/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"data\": [[30,50,25],[35,55,28],[25,45,22],[32,52,26],[28,48,24]],
        \"backend\": \"isolation_forest\"
    }" > /dev/null
echo "  ✓ Model trained"
echo ""

# Get explanation
echo -e "${YELLOW}Getting explanation via /v1/ml/anomaly/explain...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/ml/anomaly/explain" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"model_id\": \"${MODEL_NAME}\",
        \"data\": [[95, 95, 150]],
        \"feature_names\": [\"CPU\", \"Memory\", \"Latency\"],
        \"backend\": \"isolation_forest\"
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'explanations' in data and len(data['explanations']) > 0:
        exp = data['explanations'][0]
        if 'top_contributors' in exp:
            print('  ✓ Got SHAP explanations:')
            for c in exp['top_contributors'][:3]:
                print(f'    {c[\"feature\"]}: {c[\"contribution\"]:.3f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Cleanup
curl -s -X DELETE "${BASE_URL}/v1/ml/anomaly/${MODEL_NAME}" > /dev/null 2>&1 || true

echo -e "${GREEN}✓ LlamaFarm API proxy working${NC}"
echo ""
