#!/bin/bash
# SHAP Explainability Demo
# Demonstrates SHAP-based model explanations for ML predictions
#
# Features demonstrated:
# - Listing available explainers
# - Training an anomaly detection model
# - Generating SHAP explanations
# - Feature importance computation
# - Natural language narratives

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   SHAP Explainability Demo${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Configuration - Use LlamaFarm API (port 8005) for production usage
# Falls back to Universal Runtime (port 11545) for local testing
PORT="${LLAMAFARM_PORT:-${UNIVERSAL_PORT:-8005}}"
BASE_URL="${LLAMAFARM_URL:-http://localhost:${PORT}}"

# Check if server is running
echo -e "${YELLOW}Checking server status...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo

# =============================================================================
# 1. List Available Explainers
# =============================================================================
echo -e "${BLUE}1. Listing available SHAP explainers...${NC}"

EXPLAINERS=$(curl -s "${BASE_URL}/v1/explain/explainers")
echo "$EXPLAINERS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for e in data.get('explainers', []):
    print(f\"  - {e['name']}: {e['description']}\")
    print(f\"    Supports: {', '.join(e['supported_models'])}\")"
echo

# =============================================================================
# 2. Train an Anomaly Detection Model
# =============================================================================
echo -e "${BLUE}2. Training anomaly detection model for explanation...${NC}"

# Generate training data with 5 features
TRAIN_DATA=$(python3 -c "
import json
import random
random.seed(42)
data = [[random.gauss(0, 1) for _ in range(5)] for _ in range(100)]
print(json.dumps(data))")

# Fit the model
FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"data\": ${TRAIN_DATA},
        \"model_id\": \"shap-demo-model\",
        \"backend\": \"isolation_forest\",
        \"contamination\": 0.1
    }")

echo "$FIT_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Model: {data.get('model_id', 'N/A')}\")" || echo "  Model trained"
echo -e "${GREEN}✓ Model trained${NC}"
echo

# =============================================================================
# 3. Generate SHAP Explanations
# =============================================================================
echo -e "${BLUE}3. Generating SHAP explanations...${NC}"

# Create test samples (including one potential anomaly)
TEST_DATA='[
    [0.1, 0.2, -0.1, 0.3, 0.0],
    [-0.5, 0.1, 0.2, -0.3, 0.1],
    [3.5, -2.1, 4.0, -3.2, 2.8]
]'

EXPLAIN_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/explain/shap" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_type\": \"anomaly\",
        \"model_id\": \"shap-demo-model\",
        \"data\": ${TEST_DATA},
        \"feature_names\": [\"temp\", \"humidity\", \"pressure\", \"wind\", \"precip\"],
        \"top_k\": 3,
        \"generate_narrative\": true
    }")

# Check if we got explanations
if echo "$EXPLAIN_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'explanations' in data:
    exp_type = data.get('explainer_type', 'unknown')
    time_ms = data.get('explain_time_ms', 0)
    print(f'  Explainer type: {exp_type}')
    print(f'  Explain time: {time_ms:.2f}ms')
    print()
    for exp in data['explanations']:
        idx = exp['sample_index']
        base = exp['base_value']
        pred = exp['prediction']
        print(f'  Sample {idx}:')
        print(f'    Base value: {base:.4f}')
        print(f'    Prediction: {pred:.4f}')
        print(f'    Top contributors:')
        for c in exp['contributions']:
            arrow = '↑' if c['direction'] == 'increases' else '↓'
            print(f'      {c[\"feature\"]}: {c[\"value\"]:.2f} → {arrow} {c[\"shap_value\"]:+.4f}')
        print()

    if data.get('narrative'):
        print('  Narrative:')
        print(f'    {data[\"narrative\"][\"summary\"]}')
        for d in data['narrative'].get('details', [])[:3]:
            print(f'    - {d}')
    sys.exit(0)
else:
    print(f'  Response: {data}')
    sys.exit(1)" 2>/dev/null; then
    echo -e "${GREEN}✓ SHAP explanations generated${NC}"
else
    echo -e "${YELLOW}Note: SHAP explanation may require model to be cached${NC}"
    echo "  Response: $EXPLAIN_RESPONSE"
fi
echo

# =============================================================================
# 4. Feature Importance
# =============================================================================
echo -e "${BLUE}4. Computing global feature importance...${NC}"

IMPORTANCE_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/explain/importance" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_type\": \"anomaly\",
        \"model_id\": \"shap-demo-model\",
        \"data\": ${TRAIN_DATA},
        \"feature_names\": [\"temp\", \"humidity\", \"pressure\", \"wind\", \"precip\"]
    }")

if echo "$IMPORTANCE_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'importances' in data:
    time_ms = data.get('compute_time_ms', 0)
    print(f'  Compute time: {time_ms:.2f}ms')
    print()
    print('  Feature Importance Rankings:')
    for i, imp in enumerate(data['importances'], 1):
        bar_len = int(imp['importance'] * 50)
        bar = '█' * bar_len + '░' * (50 - bar_len)
        print(f'    {i}. {imp[\"feature\"]:12} {bar} {imp[\"importance\"]:.4f}')
    sys.exit(0)
else:
    sys.exit(1)" 2>/dev/null; then
    echo -e "${GREEN}✓ Feature importance computed${NC}"
else
    echo -e "${YELLOW}Note: Feature importance may require model to be cached${NC}"
fi
echo

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Demo completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "SHAP Explainability Features:"
echo "  • Tree explainer for tree-based models (IForest, CatBoost)"
echo "  • Linear explainer for linear models"
echo "  • Kernel explainer for any model (model-agnostic)"
echo "  • Natural language narrative generation"
echo "  • Global feature importance ranking"
echo
echo "API Endpoints:"
echo "  GET  /v1/explain/explainers  - List available explainers"
echo "  POST /v1/explain/shap        - Generate SHAP explanation"
echo "  POST /v1/explain/importance  - Compute feature importance"
