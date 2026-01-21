#!/bin/bash
# Demo: SHAP-based Anomaly Explanations
# Phase 15 of Universal Runtime ML Enhancements
#
# Explains WHY data points are flagged as anomalies using SHAP values.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

echo "=========================================="
echo "SHAP Anomaly Explanation Demo"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "   ERROR: LlamaFarm API not running at $BASE_URL"
    echo "   Start with: nx start server"
    exit 1
fi
echo "   Server is healthy!"
echo ""

# Generate training data (normal server metrics)
echo "2. Training an Anomaly Detection Model"
echo "   Generating normal server metrics data..."
echo ""

TRAINING_DATA=$(python3 -c "
import random
random.seed(42)
# Normal server metrics: CPU ~30%, Memory ~50%, Latency ~25ms
data = []
for _ in range(100):
    cpu = 30 + random.uniform(-10, 10)
    memory = 50 + random.uniform(-15, 15)
    latency = 25 + random.uniform(-5, 5)
    data.append([round(cpu, 1), round(memory, 1), round(latency, 1)])
print(str(data).replace(' ', ''))
")

# Train the model (overwrite=true to use consistent name)
echo "   Training IsolationForest model..."
TRAIN_RESPONSE=$(curl -s -X POST "$BASE_URL/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"server-metrics-model\",
        \"data\": $TRAINING_DATA,
        \"backend\": \"isolation_forest\",
        \"contamination\": 0.1,
        \"overwrite\": true
    }")

echo "   Training response:"
echo "$TRAIN_RESPONSE" | python3 -m json.tool
echo ""

# Extract the versioned model name from training response
MODEL_NAME=$(echo "$TRAIN_RESPONSE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('versioned_name', 'server-metrics-model'))")
echo "   Using model: $MODEL_NAME"
echo ""

# Check model status - use models list and filter
echo "3. Checking Model Status"
RESPONSE=$(curl -s "$BASE_URL/v1/anomaly/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for m in data.get('data', []):
    if m.get('base_name') == 'server-metrics-model':
        print(json.dumps(m, indent=2))
        break
else:
    print('{\"error\": \"Model not found\"}')")
echo "   Model info:"
echo "$RESPONSE"
echo ""

# Create anomalous data points
echo "4. Generating Anomalous Data Points"
echo "   Creating server metrics that should be anomalous..."
echo ""

# Different types of anomalies
ANOMALY_1="[95.0, 50.0, 25.0]"  # High CPU only
ANOMALY_2="[30.0, 95.0, 200.0]"  # High Memory + High Latency
ANOMALY_3="[95.0, 95.0, 200.0]"  # All high (severe anomaly)

echo "   Anomaly 1: CPU=95%, Memory=50%, Latency=25ms (High CPU)"
echo "   Anomaly 2: CPU=30%, Memory=95%, Latency=200ms (High Memory + Latency)"
echo "   Anomaly 3: CPU=95%, Memory=95%, Latency=200ms (All metrics high)"
echo ""

# Explain the anomalies
echo "5. Explaining Anomalies with SHAP"
echo "   Asking WHY each point is anomalous..."
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/anomaly/explain" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"$MODEL_NAME\",
        \"backend\": \"isolation_forest\",
        \"data\": [$ANOMALY_1, $ANOMALY_2, $ANOMALY_3],
        \"feature_names\": [\"cpu\", \"memory\", \"latency\"],
        \"background_samples\": 50,
        \"nsamples\": 50
    }")

echo "   SHAP Explanations:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Extract top contributors for each anomaly
echo "6. Summary: Top Contributors to Anomaly Score"
echo ""

echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
if 'explanations' in data:
    anomaly_labels = [
        'Anomaly 1 (High CPU)',
        'Anomaly 2 (High Mem+Lat)',
        'Anomaly 3 (All High)'
    ]
    for i, exp in enumerate(data['explanations']):
        print(f'   {anomaly_labels[i]}:')
        for contrib in exp.get('top_contributors', [])[:3]:
            direction = '↑' if contrib['direction'] == 'high' else '↓'
            print(f'     - {contrib[\"feature\"]}: {direction} (importance: {contrib[\"importance\"]:.3f}, value: {contrib[\"value\"]:.1f})')
        print()
"

# Score the anomalies to show correlation
echo "7. Anomaly Scores for Reference"
echo "   Scoring the same anomalous points..."
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"data\": [$ANOMALY_1, $ANOMALY_2, $ANOMALY_3]
    }")

echo "   Anomaly scores (higher = more anomalous):"
echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
labels = ['High CPU', 'High Mem+Lat', 'All High']
if 'data' in data:
    for i, item in enumerate(data['data']):
        score = item.get('score', 0)
        is_anom = item.get('is_anomaly', False)
        marker = '🚨 ANOMALY' if is_anom else '✓ Normal'
        print(f'   {labels[i]}: {score:.3f} {marker}')
elif 'scores' in data:
    for i, score in enumerate(data['scores']):
        is_anomaly = '🚨 ANOMALY' if score > 0.5 else '✓ Normal'
        print(f'   {labels[i]}: {score:.3f} {is_anomaly}')
"
echo ""

# Clean up
echo "8. Cleanup"
curl -s -X DELETE "$BASE_URL/v1/anomaly/models/$MODEL_NAME" > /dev/null
echo "   Deleted model: $MODEL_NAME"
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key insights:"
echo "  - SHAP explains feature contributions to anomaly scores"
echo "  - 'direction' shows if feature is unusually high or low"
echo "  - 'importance' shows how much each feature contributes"
echo "  - Top contributors help understand WHY something is anomalous"
echo ""
echo "Use cases:"
echo "  - Debugging ML model predictions"
echo "  - Root cause analysis for anomalies"
echo "  - Building trust in anomaly detection systems"
