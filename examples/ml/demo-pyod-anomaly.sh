#!/bin/bash
# Demo: PyOD Advanced Anomaly Detection
# Phase 17 of Universal Runtime ML Enhancements
#
# Compares COPOD, HBOS, and ECOD backends vs IsolationForest.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi
RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:${LF_RUNTIME_PORT:-11540}}"

echo "=========================================="
echo "PyOD Advanced Anomaly Detection Demo"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "$RUNTIME_URL/health" > /dev/null 2>&1; then
    echo "   ERROR: Server not running at $RUNTIME_URL"
    echo "   Start with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo "   Server is healthy!"
echo ""

# Generate training data (normal server metrics)
echo "2. Generating Training Data"
echo "   Creating normal server metrics (100 samples)..."
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

# Create anomalous test data
NORMAL_DATA="[[32.5, 52.3, 24.1], [28.7, 48.9, 26.2], [31.2, 51.1, 25.5]]"
ANOMALY_DATA="[[95.0, 95.0, 200.0], [5.0, 95.0, 5.0], [95.0, 5.0, 200.0]]"
TEST_DATA="[$NORMAL_DATA, $ANOMALY_DATA]"
TEST_DATA=$(python3 -c "print('[[32.5,52.3,24.1],[28.7,48.9,26.2],[31.2,51.1,25.5],[95.0,95.0,200.0],[5.0,95.0,5.0],[95.0,5.0,200.0]]')")

echo "   Test data:"
echo "     Normal points (first 3): CPU ~30%, Memory ~50%, Latency ~25ms"
echo "     Anomaly points (last 3): Extreme values"
echo ""

# Test different backends
echo "3. Comparing Anomaly Detection Backends"
echo "   Training and scoring with 4 different backends..."
echo ""

BACKENDS=("isolation_forest" "copod" "hbos" "ecod")
BACKEND_DESC=("IsolationForest (sklearn)" "COPOD (Copula-Based)" "HBOS (Histogram-Based)" "ECOD (Empirical CDF)")

for i in "${!BACKENDS[@]}"; do
    BACKEND="${BACKENDS[$i]}"
    DESC="${BACKEND_DESC[$i]}"
    MODEL_NAME="test-${BACKEND}"

    echo "   === ${DESC} ==="

    # Train model
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/anomaly/fit" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"data\": $TRAINING_DATA,
            \"backend\": \"$BACKEND\",
            \"contamination\": 0.1
        }")

    TRAINING_TIME=$(echo "$RESPONSE" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin).get('training_time_ms', 0):.1f}\")")
    echo "   Training time: ${TRAINING_TIME}ms"

    # Score test data
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/anomaly/score" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\",
            \"data\": $TEST_DATA,
            \"backend\": \"$BACKEND\"
        }")

    echo "   Scores:"
    echo "$RESPONSE" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
if 'data' in data:
    labels = ['Normal 1', 'Normal 2', 'Normal 3', 'Anomaly 1', 'Anomaly 2', 'Anomaly 3']
    for i, item in enumerate(data['data']):
        flag = '🚨' if item['is_anomaly'] else '✓'
        print(f'     {labels[i]}: {item[\"score\"]:.3f} {flag}')
"

    # Delete model
    curl -s -X DELETE "$RUNTIME_URL/v1/anomaly/$MODEL_NAME" > /dev/null 2>&1 || true
    echo ""
done

# Summary comparison
echo "4. Backend Characteristics"
echo ""
echo "   | Backend          | Speed   | Hyperparams | Best For                    |"
echo "   |------------------|---------|-------------|------------------------------|"
echo "   | IsolationForest  | Fast    | Few         | General purpose, default     |"
echo "   | COPOD            | Fast    | None!       | High-dimensional data        |"
echo "   | HBOS             | Fastest | Few         | Large datasets, streaming    |"
echo "   | ECOD             | Medium  | None!       | Interpretable, statistical   |"
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key insights:"
echo "  - COPOD: No hyperparameters, uses copula for dependency modeling"
echo "  - HBOS: Very fast histogram-based, good for high-throughput"
echo "  - ECOD: Uses empirical CDF, statistically interpretable"
echo "  - All backends integrate with existing /v1/anomaly/fit and /score APIs"
echo ""
echo "Use cases:"
echo "  - COPOD: When you want zero hyperparameter tuning"
echo "  - HBOS: When speed is critical (100k+ samples)"
echo "  - ECOD: When you need statistical interpretation"
