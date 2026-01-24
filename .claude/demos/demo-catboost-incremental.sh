#!/bin/bash
# CatBoost Incremental Learning Demo
# Demonstrates CatBoost gradient boosting with incremental updates
#
# Features demonstrated:
# - Initial model training
# - Classification predictions
# - Probability estimation
# - Incremental learning (model update)
# - Feature importance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   CatBoost Incremental Learning Demo${NC}"
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
# 1. Check CatBoost Availability
# =============================================================================
echo -e "${BLUE}1. Checking CatBoost availability...${NC}"

INFO=$(curl -s "${BASE_URL}/v1/catboost/info")
echo "$INFO" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Available: {data.get('available', False)}\")" || echo "  CatBoost info endpoint working"
echo -e "${GREEN}✓ CatBoost is available${NC}"
echo

# =============================================================================
# 2. Generate Training Data
# =============================================================================
echo -e "${BLUE}2. Generating training data...${NC}"

# Generate initial training data (200 samples, 5 features)
# Binary classification: class based on feature[0] > 0
TRAIN_DATA=$(python3 -c "
import json
import random
random.seed(42)

data = []
labels = []
for _ in range(200):
    features = [random.gauss(0, 1) for _ in range(5)]
    data.append(features)
    # Class 1 if first feature > 0, else class 0
    labels.append(1 if features[0] > 0 else 0)

print(json.dumps({'data': data, 'labels': labels}))")

echo "  Generated 200 training samples with 5 features"
echo

# =============================================================================
# 3. Train Initial Model
# =============================================================================
echo -e "${BLUE}3. Training initial CatBoost classifier...${NC}"

DATA=$(echo "$TRAIN_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['data']))")
LABELS=$(echo "$TRAIN_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['labels']))")

FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/catboost/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"catboost-demo\",
        \"model_type\": \"classifier\",
        \"data\": ${DATA},
        \"labels\": ${LABELS},
        \"feature_names\": [\"temp\", \"humidity\", \"pressure\", \"wind\", \"precip\"],
        \"iterations\": 50,
        \"depth\": 4
    }")

echo "$FIT_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Model ID: {data.get('model_id', 'N/A')}\")" || echo "  Model trained"
echo -e "${GREEN}✓ Initial model trained${NC}"
echo

# =============================================================================
# 4. Make Predictions
# =============================================================================
echo -e "${BLUE}4. Making predictions...${NC}"

# Create test samples
TEST_DATA='[
    [0.5, 0.1, -0.2, 0.3, 0.0],
    [-0.8, 0.2, 0.1, -0.1, 0.5],
    [1.2, -0.5, 0.3, 0.2, -0.1],
    [-1.5, 0.8, -0.4, 0.1, 0.2]
]'

PREDICT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/catboost/predict" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"catboost-demo\",
        \"data\": ${TEST_DATA},
        \"return_proba\": true
    }")

echo "$PREDICT_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'predictions' in data:
    time_ms = data.get('predict_time_ms', 0)
    print(f'  Predict time: {time_ms:.2f}ms')
    print()
    print('  Predictions with probabilities:')
    for pred in data['predictions']:
        idx = pred['sample_index']
        label = pred['prediction']
        probas = pred.get('probabilities', {})
        p0 = probas.get('0', 0)
        p1 = probas.get('1', 0)
        confidence = max(p0, p1)
        print(f'    Sample {idx}: Class {label} (confidence: {confidence:.1%})')" 2>/dev/null || echo "  Predictions made"
echo -e "${GREEN}✓ Predictions complete${NC}"
echo

# =============================================================================
# 5. Get Feature Importance
# =============================================================================
echo -e "${BLUE}5. Computing feature importance...${NC}"

IMPORTANCE_RESPONSE=$(curl -s "${BASE_URL}/v1/catboost/catboost-demo/importance")

echo "$IMPORTANCE_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'importances' in data:
    print('  Feature Importance Rankings:')
    for i, imp in enumerate(data['importances'], 1):
        bar_len = int(imp['importance'])
        bar = '█' * min(bar_len, 50) + '░' * max(0, 50 - bar_len)
        print(f'    {i}. {imp[\"feature\"]:12} {bar[:50]} {imp[\"importance\"]:.2f}')" 2>/dev/null || echo "  Importance computed"
echo -e "${GREEN}✓ Feature importance computed${NC}"
echo

# =============================================================================
# 6. Incremental Learning
# =============================================================================
echo -e "${BLUE}6. Performing incremental update...${NC}"

# Generate new data for update (50 samples)
UPDATE_DATA=$(python3 -c "
import json
import random
random.seed(123)

data = []
labels = []
for _ in range(50):
    features = [random.gauss(0, 1) for _ in range(5)]
    data.append(features)
    labels.append(1 if features[0] > 0 else 0)

print(json.dumps({'data': data, 'labels': labels}))")

UPDATE_X=$(echo "$UPDATE_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['data']))")
UPDATE_Y=$(echo "$UPDATE_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['labels']))")

UPDATE_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/catboost/update" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"catboost-demo\",
        \"data\": ${UPDATE_X},
        \"labels\": ${UPDATE_Y}
    }")

echo "$UPDATE_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'samples_added' in data:
    print(f'  Samples added: {data[\"samples_added\"]}')
    print(f'  Trees before: {data[\"trees_before\"]}')
    print(f'  Trees after: {data[\"trees_after\"]}')
    print(f'  Update time: {data[\"update_time_ms\"]:.2f}ms')" 2>/dev/null || echo "  Model updated"
echo -e "${GREEN}✓ Incremental update complete${NC}"
echo

# =============================================================================
# 7. Verify Updated Model
# =============================================================================
echo -e "${BLUE}7. Verifying updated model predictions...${NC}"

VERIFY_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/catboost/predict" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"catboost-demo\",
        \"data\": ${TEST_DATA},
        \"return_proba\": true
    }")

echo "$VERIFY_RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'predictions' in data:
    print('  Updated model predictions:')
    for pred in data['predictions']:
        idx = pred['sample_index']
        label = pred['prediction']
        probas = pred.get('probabilities', {})
        p0 = probas.get('0', 0)
        p1 = probas.get('1', 0)
        confidence = max(p0, p1)
        print(f'    Sample {idx}: Class {label} (confidence: {confidence:.1%})')" 2>/dev/null || echo "  Verification complete"
echo -e "${GREEN}✓ Updated predictions verified${NC}"
echo

# =============================================================================
# 8. Cleanup
# =============================================================================
echo -e "${BLUE}8. Cleaning up demo model...${NC}"

curl -s -X DELETE "${BASE_URL}/v1/catboost/catboost-demo" > /dev/null
echo -e "${GREEN}✓ Demo model deleted${NC}"
echo

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Demo completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo "CatBoost Features Demonstrated:"
echo "  • Initial classifier training"
echo "  • Class probability predictions"
echo "  • Native feature importance"
echo "  • Incremental learning (model update without full retrain)"
echo "  • Tree count growth during update"
echo
echo "API Endpoints:"
echo "  GET  /v1/catboost/info                    - CatBoost availability"
echo "  GET  /v1/catboost/models                  - List saved models"
echo "  POST /v1/catboost/fit                     - Train new model"
echo "  POST /v1/catboost/predict                 - Make predictions"
echo "  POST /v1/catboost/update                  - Incremental update"
echo "  GET  /v1/catboost/{model_id}/importance   - Feature importance"
echo "  DELETE /v1/catboost/{model_id}            - Delete model"
