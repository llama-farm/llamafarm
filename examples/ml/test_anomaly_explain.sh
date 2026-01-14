#!/bin/bash
# Test Anomaly SHAP Explanations endpoint
#
# This script demonstrates:
# 1. Training anomaly detector
# 2. Getting SHAP explanations for anomalies
# 3. Understanding feature contributions
#
# Usage: ./test_anomaly_explain.sh [PORT]
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
echo -e "${BLUE}  Anomaly SHAP Explanations Test${NC}"
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
# Test 1: Train Model and Get Explanations
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Train and Explain${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

MODEL_NAME="shap-test-$$"

# Generate training data (server metrics: CPU, Memory, Latency)
TRAINING_DATA=$(python3 -c "
import random
random.seed(42)
data = []
for _ in range(100):
    cpu = 30 + random.uniform(-10, 10)      # Normal: 20-40%
    memory = 50 + random.uniform(-15, 15)   # Normal: 35-65%
    latency = 25 + random.uniform(-5, 5)    # Normal: 20-30ms
    data.append([round(cpu, 1), round(memory, 1), round(latency, 1)])
print(str(data).replace(' ', ''))
")

echo -e "${YELLOW}Training anomaly detector...${NC}"
echo "  Features: CPU (%), Memory (%), Latency (ms)"
echo "  Normal ranges: CPU 20-40%, Memory 35-65%, Latency 20-30ms"
echo ""

RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"data\": ${TRAINING_DATA},
        \"backend\": \"isolation_forest\",
        \"contamination\": 0.1
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", 0)} samples')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Get explanations for different scenarios
echo -e "${YELLOW}Getting SHAP explanations for test cases...${NC}"
echo ""

# Test cases
declare -a TEST_CASES=(
    "High CPU|95,50,25"
    "High Memory|30,95,25"
    "High Latency|30,50,150"
    "All High|90,90,150"
    "Normal|30,50,25"
)

for CASE in "${TEST_CASES[@]}"; do
    IFS='|' read -r DESC VALUES <<< "$CASE"
    DATA="[[${VALUES}]]"

    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/explain" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"model_id\": \"${MODEL_NAME}\",
            \"data\": ${DATA},
            \"feature_names\": [\"CPU\", \"Memory\", \"Latency\"],
            \"backend\": \"isolation_forest\"
        }")

    echo -e "${CYAN}  ${DESC} [${VALUES}]:${NC}"
    echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'explanations' in data and len(data['explanations']) > 0:
        exp = data['explanations'][0]
        if 'top_contributors' in exp:
            contributors = exp['top_contributors']
            for c in contributors:
                bar = '█' * int(abs(c['contribution']) * 20)
                sign = '+' if c['contribution'] > 0 else '-'
                print(f'    {c[\"feature\"]:<10}: {sign}{abs(c[\"contribution\"]):.3f} {bar}')
        else:
            print('    No contributors found')
    else:
        print(f'    Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    echo ""
done

# Cleanup
curl -s -X DELETE "${BASE_URL}/v1/anomaly/${MODEL_NAME}" > /dev/null 2>&1 || true

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoint: POST /v1/anomaly/explain"
echo ""
echo "Request format:"
echo "  {"
echo "    \"model_id\": \"my-model\","
echo "    \"data\": [[val1, val2, ...]],"
echo "    \"feature_names\": [\"feat1\", \"feat2\", ...],"
echo "    \"backend\": \"isolation_forest\""
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"explanations\": ["
echo "      {"
echo "        \"features\": {\"CPU\": 0.5, \"Memory\": 0.3},"
echo "        \"top_contributors\": ["
echo "          {\"feature\": \"CPU\", \"contribution\": 0.5, \"value\": 95}"
echo "        ]"
echo "      }"
echo "    ]"
echo "  }"
echo ""
echo "Interpretation:"
echo "  - Positive contribution → feature makes point MORE anomalous"
echo "  - Negative contribution → feature makes point LESS anomalous"
echo "  - Higher absolute value → stronger influence"
echo ""
