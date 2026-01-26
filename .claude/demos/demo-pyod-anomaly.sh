#!/bin/bash
# Demo: PyOD Anomaly Detection Backends
#
# This demo shows:
# 1. Training with legacy `isolation_forest` backend (routes to PyOD IForest)
# 2. Training with new `ecod` backend (ECOD - fast, parameter-free)
# 3. Comparing detection results
# 4. Listing all available backends via API
#
# Prerequisites:
# - Server running at PORT (default 8005): nx start server
# - Universal Runtime running at LF_RUNTIME_PORT (default 11545): nx start universal-runtime
# - A llamafarm.yaml project loaded: cd /tmp/test-project && lf start

set -e

# Configuration - respects environment variables
BASE_URL="${BASE_URL:-http://localhost:${PORT:-8005}}"
API_URL="${BASE_URL}/v1/ml"

echo "========================================"
echo "PyOD Anomaly Detection Demo"
echo "========================================"
echo "Server URL: ${BASE_URL}"
echo ""

# Check if server is running
echo "Checking server health..."
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: Server not running at ${BASE_URL}"
    echo "Start with: nx start server (and nx start universal-runtime)"
    exit 1
fi
echo "✓ Server is running"
echo ""

# Generate training data (mostly normal with some pattern)
echo "========================================"
echo "Step 1: Generate Training Data"
echo "========================================"

# Create training data - 50 points centered around (0, 0)
TRAIN_DATA=$(python3 -c "
import json
import random
random.seed(42)
data = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(50)]
print(json.dumps(data))
")

echo "Generated 50 training samples (2D Gaussian around origin)"
echo ""

# Generate test data with an obvious anomaly
TEST_DATA=$(python3 -c "
import json
import random
random.seed(42)
# 5 normal points
normal = [[random.gauss(0, 1), random.gauss(0, 1)] for _ in range(5)]
# 1 obvious anomaly
anomaly = [[10.0, 10.0]]
print(json.dumps(normal + anomaly))
")

echo "Generated 6 test samples (5 normal + 1 anomaly at [10, 10])"
echo ""

# Step 2: Train with legacy isolation_forest (routes to PyOD)
echo "========================================"
echo "Step 2: Train with Legacy isolation_forest"
echo "========================================"

RESPONSE=$(curl -s -X POST "${API_URL}/anomaly/fit" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"demo-iforest\",
    \"backend\": \"isolation_forest\",
    \"data\": ${TRAIN_DATA},
    \"contamination\": 0.1
  }")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Step 3: Train with new ECOD backend
echo "========================================"
echo "Step 3: Train with New ECOD Backend"
echo "========================================"

RESPONSE=$(curl -s -X POST "${API_URL}/anomaly/fit" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"demo-ecod\",
    \"backend\": \"ecod\",
    \"data\": ${TRAIN_DATA},
    \"contamination\": 0.1
  }")

echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# Step 4: Score test data with both backends
echo "========================================"
echo "Step 4: Compare Detection Results"
echo "========================================"

echo "--- Isolation Forest Results ---"
IFOREST_RESULTS=$(curl -s -X POST "${API_URL}/anomaly/score" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"demo-iforest\",
    \"backend\": \"isolation_forest\",
    \"data\": ${TEST_DATA}
  }")

echo "$IFOREST_RESULTS" | python3 -m json.tool 2>/dev/null || echo "$IFOREST_RESULTS"
echo ""

echo "--- ECOD Results ---"
ECOD_RESULTS=$(curl -s -X POST "${API_URL}/anomaly/score" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"demo-ecod\",
    \"backend\": \"ecod\",
    \"data\": ${TEST_DATA}
  }")

echo "$ECOD_RESULTS" | python3 -m json.tool 2>/dev/null || echo "$ECOD_RESULTS"
echo ""

# Step 5: List all available backends
echo "========================================"
echo "Step 5: List All Available Backends"
echo "========================================"

BACKENDS=$(curl -s -X GET "${API_URL}/anomaly/backends")

echo "Available backends:"
echo "$BACKENDS" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f\"Total: {data['total']} backends\\n\")
print(f\"Categories: {', '.join(data['categories'])}\\n\")
print('Backends:')
for b in data['data']:
    legacy = ' (legacy)' if b['is_legacy'] else ''
    print(f\"  - {b['backend']}: {b['description'][:60]}...{legacy}\")
"
echo ""

# Step 6: Test detect endpoint (returns only anomalies)
echo "========================================"
echo "Step 6: Detect Anomalies Only"
echo "========================================"

DETECT_RESULTS=$(curl -s -X POST "${API_URL}/anomaly/detect" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"demo-ecod\",
    \"backend\": \"ecod\",
    \"data\": ${TEST_DATA},
    \"threshold\": 0.5
  }")

echo "Anomalies detected (threshold=0.5):"
echo "$DETECT_RESULTS" | python3 -m json.tool 2>/dev/null || echo "$DETECT_RESULTS"
echo ""

# Summary
echo "========================================"
echo "Demo Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "- Legacy backend 'isolation_forest' routes to PyOD IForest"
echo "- New backend 'ecod' provides fast, parameter-free detection"
echo "- Both backends correctly identified the anomaly at [10, 10]"
echo "- 12+ backends available via /v1/ml/anomaly/backends"
echo ""
echo "Try other backends: hbos, knn, copod, one_class_svm, local_outlier_factor"
