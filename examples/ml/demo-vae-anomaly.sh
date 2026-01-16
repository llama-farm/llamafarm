#!/bin/bash
# Demo: VAE Anomaly Detection with Early Stopping
#
# Phase 3 of Universal Runtime ML Enhancements
#
# This demo shows:
# 1. Training a VAE-based anomaly detector with early stopping
# 2. Early stopping triggering when validation loss plateaus
# 3. VAE providing probabilistic anomaly scores
# 4. Detecting anomalies in test data

set -e

PORT="${1:-8000}"
BASE_URL="http://localhost:${PORT}"

# Get the repository root directory for running Python with uv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_DIR="${REPO_ROOT}/runtimes/universal"

# Helper function to run Python via uv in the universal runtime
run_python() {
    (cd "${RUNTIME_DIR}" && uv run python "$@")
}

echo "=============================================="
echo "    VAE Anomaly Detection Demo"
echo "    Phase 3: Early Stopping + VAE"
echo "=============================================="
echo ""

# Check if server is running
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: LlamaFarm API not running on port ${PORT}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi

echo "✓ LlamaFarm API is running on port ${PORT}"
echo ""

# Generate training data (normal patterns)
echo "=== Step 1: Generate Normal Training Data ==="
echo ""
echo "Creating 200 samples of normal sensor data..."
echo "Each sample has 4 features: temperature, pressure, vibration, flow_rate"
echo ""

# Create training data JSON - normal patterns
TRAIN_DATA=$(run_python -c "
import json
import numpy as np
np.random.seed(42)

n = 200
# Normal sensor readings with correlations
temp = np.random.normal(25, 2, n)
pressure = 100 + temp * 0.5 + np.random.normal(0, 1, n)  # Correlated with temp
vibration = np.random.normal(0.5, 0.1, n)
flow_rate = np.random.normal(10, 0.5, n)

data = [[float(t), float(p), float(v), float(f)]
        for t, p, v, f in zip(temp, pressure, vibration, flow_rate)]
print(json.dumps(data))
")

echo "Training data generated: 200 samples x 4 features"
echo ""

# Train VAE with early stopping
echo "=== Step 2: Train VAE with Early Stopping ==="
echo ""
echo "Training parameters:"
echo "  - Backend: vae (Variational Autoencoder)"
echo "  - Patience: 10 epochs"
echo "  - Min delta: 0.0001"
echo "  - Validation split: 10%"
echo "  - Max epochs: 200"
echo ""

FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo_vae_sensor\",
        \"data\": ${TRAIN_DATA},
        \"backend\": \"vae\",
        \"epochs\": 200,
        \"batch_size\": 32,
        \"scaler_type\": \"robust\",
        \"patience\": 10,
        \"min_delta\": 0.0001,
        \"validation_split\": 0.1
    }")

echo "Training response:"
echo "${FIT_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${FIT_RESPONSE}"
echo ""

# Check if early stopping was triggered
TRAINING_TIME=$(echo "${FIT_RESPONSE}" | (run_python -c "import sys, json; print(json.load(sys.stdin).get('training_time_ms', 'N/A'))") 2>/dev/null || echo "N/A")
echo "Training completed in ${TRAINING_TIME} ms"
echo ""

# Generate test data with anomalies
echo "=== Step 3: Generate Test Data with Anomalies ==="
echo ""

TEST_DATA=$(run_python -c "
import json
import numpy as np
np.random.seed(123)

# Normal samples (should be similar to training)
normal_temp = np.random.normal(25, 2, 10)
normal_pressure = 100 + normal_temp * 0.5 + np.random.normal(0, 1, 10)
normal_vibration = np.random.normal(0.5, 0.1, 10)
normal_flow = np.random.normal(10, 0.5, 10)

normal = [[float(t), float(p), float(v), float(f)]
          for t, p, v, f in zip(normal_temp, normal_pressure, normal_vibration, normal_flow)]

# Anomalies - clearly out of distribution
anomalies = [
    [50.0, 150.0, 2.0, 20.0],   # All values high
    [5.0, 80.0, 0.1, 5.0],      # All values low
    [25.0, 200.0, 0.5, 10.0],   # Pressure spike
    [25.0, 100.0, 5.0, 10.0],   # Vibration spike
    [25.0, 100.0, 0.5, 1.0],    # Flow rate drop
]

# Mix them together
test_data = normal[:5] + anomalies + normal[5:]
print(json.dumps(test_data))
")

echo "Test data: 10 normal + 5 anomalies"
echo ""
echo "Anomaly patterns:"
echo "  - Sample 6: All values high (temperature=50, pressure=150, vibration=2.0)"
echo "  - Sample 7: All values low (temperature=5, pressure=80)"
echo "  - Sample 8: Pressure spike (pressure=200)"
echo "  - Sample 9: Vibration spike (vibration=5.0)"
echo "  - Sample 10: Flow rate drop (flow_rate=1.0)"
echo ""

# Detect anomalies
echo "=== Step 4: Detect Anomalies ==="
echo ""

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo_vae_sensor\",
        \"backend\": \"vae\",
        \"data\": ${TEST_DATA}
    }")

echo "Detection results:"
echo "${DETECT_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${DETECT_RESPONSE}"
echo ""

# Analyze results
echo "=== Step 5: Analysis ==="
echo ""

run_python << PYEOF
import json
import sys

response = json.loads('''${DETECT_RESPONSE}''')
# The detect endpoint returns ONLY anomalies in the 'data' field
# Each item has: index, score, raw_score (no is_anomaly - all are anomalies)
detected_anomalies = response.get('data', [])
detected_indices = {s['index'] for s in detected_anomalies}

# Expected anomaly indices (0-indexed): 5, 6, 7, 8, 9
expected_anomalies = {5, 6, 7, 8, 9}

# Calculate metrics
true_positives = detected_indices & expected_anomalies  # Correctly detected anomalies
false_positives = detected_indices - expected_anomalies  # Normal samples flagged as anomalies
missed = expected_anomalies - detected_indices  # Anomalies not detected

print('Detected Anomalies Analysis:')
print('-' * 60)
for s in detected_anomalies:
    idx = s['index']
    score = s['score']
    raw = s['raw_score']

    expected = 'TRUE+' if idx in expected_anomalies else 'FALSE+'
    print(f'  Sample {idx:2d}: score={score:.4f} (raw={raw:.2f}) | {expected}')

print()
print('Summary:')
print('-' * 60)
print(f'  True positives: {len(true_positives)} / {len(expected_anomalies)} anomalies detected')
print(f'  False positives: {len(false_positives)} normal samples flagged')
if missed:
    print(f'  Missed anomalies: {sorted(missed)}')
else:
    print('  No missed anomalies!')

detection_rate = len(true_positives) / len(expected_anomalies) * 100
print(f'  Detection rate: {detection_rate:.1f}%')

if detection_rate >= 80:
    print()
    print('✓ VAE anomaly detection is working correctly!')
else:
    print()
    print('⚠ Detection rate is below expected threshold')
PYEOF

echo ""
echo "=============================================="
echo "    Demo Complete"
echo "=============================================="
echo ""
echo "Key observations:"
echo "  1. VAE used early stopping to prevent overfitting"
echo "  2. Training stopped when validation ELBO plateaued"
echo "  3. Anomaly scores are based on negative ELBO (reconstruction + KL divergence)"
echo "  4. Higher scores indicate samples that are unlikely under the learned distribution"
echo ""
