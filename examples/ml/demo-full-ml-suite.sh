#!/bin/bash
# Demo: Full ML Suite Integration
# Phase 19 of Universal Runtime ML Enhancements
#
# Demonstrates all ML features working together in an end-to-end workflow.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

echo "==========================================="
echo "Full ML Suite Integration Demo"
echo "==========================================="
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

# ============================================
# ANOMALY DETECTION PIPELINE
# ============================================
echo "==========================================="
echo "2. Anomaly Detection Pipeline"
echo "==========================================="
echo ""

echo "   a) Training IsolationForest model..."
TRAINING_DATA=$(python3 -c "
import random
random.seed(42)
data = []
for _ in range(100):
    cpu = 30 + random.uniform(-10, 10)
    memory = 50 + random.uniform(-15, 15)
    latency = 25 + random.uniform(-5, 5)
    data.append([round(cpu, 1), round(memory, 1), round(latency, 1)])
print(str(data).replace(' ', ''))
")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-pipeline\",
        \"data\": $TRAINING_DATA,
        \"backend\": \"isolation_forest\",
        \"contamination\": 0.1
    }")

TRAINING_TIME=$(echo "$RESPONSE" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('training_time_ms', 0):.1f}\")")
echo "      Training time: ${TRAINING_TIME}ms"

echo "   b) Scoring test data..."
TEST_DATA="[[32.5,52.3,24.1],[28.7,48.9,26.2],[95.0,95.0,200.0]]"
RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/anomaly/score" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-pipeline\",
        \"data\": $TEST_DATA,
        \"backend\": \"isolation_forest\"
    }")

echo "      Scores:"
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'data' in data:
    labels = ['Normal 1', 'Normal 2', 'Anomaly']
    for i, item in enumerate(data['data']):
        flag = '🚨' if item['is_anomaly'] else '✓'
        print(f'        {labels[i]}: {item[\"score\"]:.3f} {flag}')
"

echo "   c) Getting SHAP explanations..."
RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/anomaly/explain" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"demo-pipeline\",
        \"data\": [[95.0, 95.0, 200.0]],
        \"feature_names\": [\"cpu\", \"memory\", \"latency\"],
        \"backend\": \"isolation_forest\"
    }")

echo "      Top contributors for anomaly:"
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'explanations' in data and len(data['explanations']) > 0:
    exp = data['explanations'][0]
    if 'top_contributors' in exp:
        for c in exp['top_contributors'][:3]:
            print(f'        {c[\"feature\"]}: {c[\"contribution\"]:.3f}')
"
echo ""

# ============================================
# DATASET QUALITY AUDIT
# ============================================
echo "==========================================="
echo "3. Dataset Quality Audit"
echo "==========================================="
echo ""

echo "   Auditing dataset with potential label errors..."
AUDIT_DATA=$(python3 -c "
import json
import random
random.seed(42)

# True labels
true_labels = [random.randint(0, 2) for _ in range(50)]

# Create prediction probabilities
pred_probs = []
for label in true_labels:
    probs = [0.05, 0.05, 0.05]
    probs[label] = 0.85 + random.uniform(0, 0.1)
    remaining = 1 - probs[label]
    for j in range(3):
        if j != label:
            probs[j] = remaining / 2
    pred_probs.append([round(p, 3) for p in probs])

# Add label noise (flip 20% of labels)
labels = true_labels.copy()
for i in range(0, 50, 5):
    wrong = [l for l in range(3) if l != true_labels[i]]
    labels[i] = random.choice(wrong)

print(json.dumps({'labels': labels, 'pred_probs': pred_probs}))
")

LABELS=$(echo "$AUDIT_DATA" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['labels']))")
PROBS=$(echo "$AUDIT_DATA" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin)['pred_probs']))")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/dataset-audit" \
    -H "Content-Type: application/json" \
    -d "{
        \"labels\": $LABELS,
        \"pred_probs\": $PROBS,
        \"label_names\": [\"cat\", \"dog\", \"bird\"]
    }")

echo "   Audit results:"
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'summary' in data:
    summary = data['summary']
    print(f'     Total samples: {summary.get(\"total_samples\", 0)}')
    print(f'     Label issues found: {summary.get(\"num_label_issues\", 0)}')
    print(f'     Overall quality: {summary.get(\"overall_quality_score\", 0):.2%}')
"
echo ""

# ============================================
# DRIFT DETECTION
# ============================================
echo "==========================================="
echo "4. Batch Drift Detection"
echo "==========================================="
echo ""

echo "   Generating stream with drift..."
DRIFT_STREAM=$(python3 -c "
import random
random.seed(42)
# First 50 values around mean 5
stream1 = [5 + random.uniform(-0.5, 0.5) for _ in range(50)]
# Next 50 values around mean 10 (drift!)
stream2 = [10 + random.uniform(-0.5, 0.5) for _ in range(50)]
print(str(stream1 + stream2).replace(' ', ''))
")

echo "   Stream: 50 values ~5, then 50 values ~10 (drift at ~50)"
echo ""

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/drift" \
    -H "Content-Type: application/json" \
    -d "{
        \"values\": $DRIFT_STREAM,
        \"algorithm\": \"adwin\",
        \"delta\": 0.001
    }")

echo "   Drift detection result:"
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
detected = data.get('drift_detected', False)
points = data.get('drift_points', [])
print(f'     Drift detected: {detected}')
if points:
    print(f'     Drift points: {points}')
"
echo ""

# ============================================
# CHANGE POINT DETECTION
# ============================================
echo "==========================================="
echo "5. Change Point Detection"
echo "==========================================="
echo ""

echo "   Generating time series with change point..."
TIMESERIES=$(python3 -c "
import random
random.seed(42)
# Segment 1: mean=5
seg1 = [5 + random.uniform(-0.5, 0.5) for _ in range(50)]
# Segment 2: mean=15
seg2 = [15 + random.uniform(-0.5, 0.5) for _ in range(50)]
data = seg1 + seg2
print(str(data).replace(' ', ''))
")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/timeseries/changepoints" \
    -H "Content-Type: application/json" \
    -d "{
        \"data\": $TIMESERIES,
        \"algorithm\": \"binseg\",
        \"n_changepoints\": 1
    }")

echo "   Results:"
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'change_points' in data:
    cps = data['change_points']
    print(f'     Change points detected: {cps}')
    print(f'     (Expected around position 50)')
"
echo ""

# ============================================
# CLEANUP
# ============================================
echo "==========================================="
echo "6. Cleanup"
echo "==========================================="
echo ""

echo "   Deleting test models and detectors..."
curl -s -X DELETE "$BASE_URL/v1/ml/anomaly/demo-pipeline" > /dev/null 2>&1 || true
curl -s -X DELETE "$BASE_URL/v1/ml/analysis/drift/demo-drift" > /dev/null 2>&1 || true
echo "   Cleanup complete!"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "==========================================="
echo "Demo Complete!"
echo "==========================================="
echo ""
echo "Features demonstrated:"
echo "  ✓ Anomaly Detection (train → score → explain)"
echo "  ✓ SHAP Explanations for anomalies"
echo "  ✓ Dataset Quality Audit with Cleanlab"
echo "  ✓ Streaming Drift Detection with River"
echo "  ✓ Change Point Detection with Ruptures"
echo ""
echo "All ML features are working together!"
echo ""
