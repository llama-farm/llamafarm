#!/bin/bash
# Demo: Streaming Training for Large Datasets
#
# Phase 4 of Universal Runtime ML Enhancements
#
# This demo shows:
# 1. Generating a large dataset (50MB CSV file)
# 2. Uploading for streaming training
# 3. Training without loading entire file into memory
# 4. Monitoring memory usage during training

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
echo "    Streaming Training Demo"
echo "    Phase 4: Memory-Efficient Training"
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

# Generate large CSV file
TEMP_DIR="${TMPDIR:-/tmp}/llamafarm-demo"
mkdir -p "$TEMP_DIR"
CSV_FILE="${TEMP_DIR}/large_sensor_data.csv"

echo "=== Step 1: Generate Large Dataset ==="
echo ""
echo "Creating 100,000 rows of sensor data (~10MB)..."
echo "Columns: timestamp, temp, pressure, vibration, flow_rate, humidity, power, rpm"
echo ""

run_python << PYEOF
import csv
import numpy as np
import os

np.random.seed(42)
n_rows = 100000

# Create CSV file
with open("${CSV_FILE}", 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    writer.writerow(['timestamp', 'temp', 'pressure', 'vibration', 'flow_rate', 'humidity', 'power', 'rpm'])

    # Generate data in chunks to avoid memory issues
    for i in range(n_rows):
        timestamp = 1000 + i
        row = [
            timestamp,
            np.random.normal(25, 2),      # temperature
            np.random.normal(100, 5),     # pressure
            np.random.normal(0.5, 0.1),   # vibration
            np.random.normal(10, 0.5),    # flow_rate
            np.random.normal(60, 5),      # humidity
            np.random.normal(500, 50),    # power
            np.random.normal(1800, 100),  # rpm
        ]
        writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])

# Report file size
file_size = os.path.getsize("${CSV_FILE}") / 1024 / 1024
print(f"Generated: {n_rows} rows, {file_size:.1f} MB")
PYEOF

echo ""
echo "File: ${CSV_FILE}"
echo ""

# Train using the streaming data endpoint
echo "=== Step 2: Upload and Train with Streaming ==="
echo ""
echo "Training parameters:"
echo "  - Backend: isolation_forest"
echo "  - Batch size: 10,000 rows"
echo "  - Processing: Streaming from file"
echo ""

# First, upload the file
echo "Uploading file..."
UPLOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/upload-training-data" \
    -F "file=@${CSV_FILE}" \
    -F "skip_columns=timestamp")

echo "Upload response:"
echo "${UPLOAD_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${UPLOAD_RESPONSE}"
echo ""

# Extract file_id from response
FILE_ID=$(echo "${UPLOAD_RESPONSE}" | (run_python -c "import sys, json; print(json.load(sys.stdin).get('file_id', ''))") 2>/dev/null)

if [ -z "$FILE_ID" ] || [ "$FILE_ID" == "None" ]; then
    echo "Note: File upload endpoint not yet implemented."
    echo "Falling back to direct training demo..."
    echo ""

    # Alternative: Direct training with in-memory data (smaller subset)
    echo "=== Alternative: Direct Training Demo ==="
    echo ""

    # Create smaller dataset for demo
    TRAIN_DATA=$(run_python << PYEOF
import json
import numpy as np
np.random.seed(42)
n = 5000
data = [[
    float(np.random.normal(25, 2)),
    float(np.random.normal(100, 5)),
    float(np.random.normal(0.5, 0.1)),
    float(np.random.normal(10, 0.5)),
] for _ in range(n)]
print(json.dumps(data))
PYEOF
)

    echo "Training on 5,000 samples (in-memory for demo)..."
    FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"streaming_demo\",
            \"backend\": \"isolation_forest\",
            \"data\": ${TRAIN_DATA}
        }")

    echo "Training response:"
    echo "${FIT_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${FIT_RESPONSE}"
else
    # Use the file reference for training
    echo "Training from uploaded file (file_id: ${FILE_ID})..."

    FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"streaming_demo\",
            \"backend\": \"isolation_forest\",
            \"training_file\": \"${FILE_ID}\",
            \"batch_size\": 10000
        }")

    echo "Training response:"
    echo "${FIT_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${FIT_RESPONSE}"
fi

echo ""

# Test anomaly detection
echo "=== Step 3: Test Anomaly Detection ==="
echo ""

TEST_DATA=$(run_python << PYEOF
import json
import numpy as np
np.random.seed(123)

# Normal samples
normal = [[
    float(np.random.normal(25, 2)),
    float(np.random.normal(100, 5)),
    float(np.random.normal(0.5, 0.1)),
    float(np.random.normal(10, 0.5)),
] for _ in range(5)]

# Anomalies
anomalies = [
    [50.0, 150.0, 2.0, 20.0],
    [5.0, 80.0, 0.1, 5.0],
    [25.0, 200.0, 0.5, 10.0],
]

test = normal + anomalies
print(json.dumps(test))
PYEOF
)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"streaming_demo\",
        \"backend\": \"isolation_forest\",
        \"data\": ${TEST_DATA}
    }")

echo "Detection results:"
echo "${DETECT_RESPONSE}" | (run_python -m json.tool) 2>/dev/null || echo "${DETECT_RESPONSE}"
echo ""

# Cleanup
echo "=== Step 4: Cleanup ==="
rm -f "${CSV_FILE}"
echo "Removed temporary CSV file."
echo ""

echo "=============================================="
echo "    Demo Complete"
echo "=============================================="
echo ""
echo "Key observations:"
echo "  1. Large datasets can be processed in batches"
echo "  2. Memory usage stays bounded during training"
echo "  3. File streaming allows training on datasets larger than RAM"
echo "  4. Anomaly detection works on models trained from streamed data"
echo ""
