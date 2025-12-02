#!/bin/bash
# Test Anomaly Detection endpoint
#
# This script demonstrates:
# 1. Fitting an anomaly detector on normal data
# 2. Saving the trained model (production-ready)
# 3. Scoring new data for anomalies
# 4. Detecting anomalies with threshold
# 5. Loading a saved model (simulating server restart)
#
# Usage: ./test_anomaly.sh [PORT]
#   PORT defaults to 11540 (Universal Runtime)

set -e

PORT=${1:-11540}
BASE_URL="http://localhost:${PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Universal Runtime Anomaly Detection Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# ============================================================================
# Test 1: Fit Anomaly Detector
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Fit Anomaly Detector${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training isolation forest on normal sensor data...${NC}"
echo "   Backend: isolation_forest"
echo "   Simulating: Temperature readings (normal range: 20-25°C)"
echo ""

# Generate normal data (temperatures between 20-25)
FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector",
        "backend": "isolation_forest",
        "data": [
            [22.1], [23.5], [21.8], [24.2], [22.7],
            [23.1], [21.5], [24.8], [22.3], [23.9],
            [21.2], [24.5], [22.8], [23.2], [21.9],
            [24.1], [22.5], [23.7], [21.6], [24.3],
            [22.2], [23.4], [21.7], [24.6], [22.9],
            [23.0], [21.4], [24.4], [22.6], [23.8]
        ],
        "contamination": 0.05
    }')

echo "Fit Response:"
echo "$FIT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$FIT_RESPONSE"
echo ""

if echo "$FIT_RESPONSE" | grep -q '"samples_fitted"'; then
    echo -e "${GREEN}✓ Anomaly detector trained successfully!${NC}"
else
    echo -e "${YELLOW}Fit may have failed${NC}"
fi
echo ""

# ============================================================================
# Test 2: Save Model (Production Workflow)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Save Model for Production${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Saving trained model to disk...${NC}"
echo "   This allows the model to persist across server restarts"
echo ""

SAVE_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/save" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector",
        "backend": "isolation_forest"
    }')

echo "Save Response:"
echo "$SAVE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$SAVE_RESPONSE"
echo ""

if echo "$SAVE_RESPONSE" | grep -q '"saved"'; then
    echo -e "${GREEN}✓ Model saved to disk!${NC}"
else
    echo -e "${YELLOW}Save may have failed${NC}"
fi
echo ""

# ============================================================================
# Test 3: List Saved Models
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: List Saved Models${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Listing available models...${NC}"

LIST_RESPONSE=$(curl -s -X GET "${BASE_URL}/v1/anomaly/models")

echo "Models Response:"
echo "$LIST_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LIST_RESPONSE"
echo ""

# ============================================================================
# Test 4: Detect Anomalies
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 4: Detect Anomalies${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Detecting anomalies in batch data...${NC}"
echo "   Threshold: 0.5 (normalized score)"
echo ""

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector",
        "backend": "isolation_forest",
        "data": [
            [22.0], [23.5], [0.0], [21.5], [100.0],
            [24.0], [-10.0], [22.8], [35.0], [23.2]
        ],
        "threshold": 0.5
    }')

echo "Detect Response:"
echo "$DETECT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DETECT_RESPONSE"
echo ""

if echo "$DETECT_RESPONSE" | grep -q '"anomalies_detected"'; then
    echo -e "${GREEN}✓ Anomaly detection completed!${NC}"
    echo ""
    echo "Summary:"
    echo "$DETECT_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    readings = [22.0, 23.5, 0.0, 21.5, 100.0, 24.0, -10.0, 22.8, 35.0, 23.2]
    anomalies = data.get('data', [])
    summary = data.get('summary', {})
    print(f'  Anomalies found: {summary.get(\"anomalies_detected\", len(anomalies))}')
    print(f'  Threshold: {summary.get(\"threshold\", \"N/A\")}')
    if anomalies:
        print(f'  Anomalous readings:')
        for a in anomalies:
            idx = a['index']
            print(f'    - {readings[idx]}°C (score: {a[\"score\"]:.4f})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${YELLOW}Detection may have failed${NC}"
    echo "$DETECT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$DETECT_RESPONSE"
fi
echo ""

# ============================================================================
# Test 5: Real API Log Anomaly Detection (from CSV files)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 5: Real API Log Anomaly Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

SCRIPT_DIR="$(dirname "$0")"
TRAINING_CSV="${SCRIPT_DIR}/api_logs_training_normal.csv"
TEST_CSV="${SCRIPT_DIR}/api_logs_test_with_anomalies.csv"

# Check if CSV files exist
if [ ! -f "$TRAINING_CSV" ] || [ ! -f "$TEST_CSV" ]; then
    echo -e "${RED}Error: CSV files not found${NC}"
    echo "   Expected: $TRAINING_CSV"
    echo "   Expected: $TEST_CSV"
else
    echo -e "${YELLOW}Training on real API logs from CSV...${NC}"
    echo "   Source: api_logs_training_normal.csv (200 normal requests)"
    echo "   Features: response_time_ms, bytes_transferred, requests_in_session"
    echo ""

    # Extract training data from CSV (columns: response_time_ms=7, bytes_transferred=8, requests_in_session=12)
    # Skip header, exclude is_anomaly column
    TRAINING_DATA=$(python3 -c "
import csv
import json
data = []
with open(\"$TRAINING_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append([
            int(row['response_time_ms']),
            int(row['bytes_transferred']),
            int(row['requests_in_session'])
        ])
print(json.dumps(data))
" 2>/dev/null)

    # Train the model
    FIT_API_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"model\": \"api_log_detector\",
            \"backend\": \"isolation_forest\",
            \"data\": ${TRAINING_DATA},
            \"contamination\": 0.01
        }")

    echo "$FIT_API_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", \"?\")} samples in {data.get(\"training_time_ms\", 0):.1f}ms')
    print(f'  Threshold: {data.get(\"model_params\", {}).get(\"threshold\", \"N/A\"):.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    echo ""

    echo -e "${YELLOW}Detecting anomalies in test data...${NC}"
    echo "   Source: api_logs_test_with_anomalies.csv (200 requests, includes real attacks)"
    echo ""

    # Extract test data from CSV (same columns, exclude is_anomaly)
    TEST_DATA=$(python3 -c "
import csv
import json
data = []
with open(\"$TEST_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append([
            int(row['response_time_ms']),
            int(row['bytes_transferred']),
            int(row['requests_in_session'])
        ])
print(json.dumps(data))
" 2>/dev/null)

    # Detect anomalies (threshold 0.75 - tuned to minimize false positives while catching real attacks)
    DETECT_API_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"model\": \"api_log_detector\",
            \"backend\": \"isolation_forest\",
            \"data\": ${TEST_DATA},
            \"threshold\": 0.75
        }")

    echo "API Log Analysis Results:"
    python3 -c "
import sys, json, csv

# Load test CSV for context
test_rows = []
with open(\"$TEST_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_rows.append(row)

# Parse detection response
response = json.loads('''${DETECT_API_RESPONSE}''')
anomaly_indices = {a['index'] for a in response.get('data', [])}
anomaly_scores = {a['index']: a['score'] for a in response.get('data', [])}

print('  ' + '-' * 100)
print(f'  {\"#\":<4} {\"Time(ms)\":<10} {\"Bytes\":<12} {\"Sessions\":<10} {\"Status\":<12} Endpoint')
print('  ' + '-' * 100)

# Only show first 20 and any anomalies
shown = 0
for i, row in enumerate(test_rows):
    is_anomaly = i in anomaly_indices
    if shown < 10 or is_anomaly or i >= len(test_rows) - 5:
        status = f'🚨 ANOMALY ({anomaly_scores.get(i, 0):.2f})' if is_anomaly else '✓ Normal'
        time_ms = row['response_time_ms']
        bytes_t = row['bytes_transferred']
        sessions = row['requests_in_session']
        endpoint = row['endpoint'][:40]
        print(f'  {i:<4} {time_ms:<10} {bytes_t:<12} {sessions:<10} {status:<12} {endpoint}')
        shown += 1
    elif shown == 10:
        print(f'  ... ({len(test_rows) - 15} more normal entries) ...')
        shown += 1

print('  ' + '-' * 100)
summary = response.get('summary', {})
print(f'  Total anomalies detected: {summary.get(\"anomalies_detected\", len(anomaly_indices))} out of {len(test_rows)} requests')

# Show details of detected anomalies
if anomaly_indices:
    print('')
    print('  Anomaly Details:')
    for idx in sorted(anomaly_indices):
        row = test_rows[idx]
        score = anomaly_scores.get(idx, 0)
        print(f'    Row {idx}: {row[\"endpoint\"]} - {row[\"response_time_ms\"]}ms, {row[\"bytes_transferred\"]} bytes, {row[\"requests_in_session\"]} reqs (score: {score:.4f})')
        # Check if this matches expected anomaly (is_anomaly column)
        if row.get('is_anomaly') == '1':
            print(f'      ✓ CONFIRMED: This is a known attack in the ground truth data')
" 2>/dev/null
fi
echo ""

# ============================================================================
# Test 5b: Mixed Data with Categorical Features (NEW)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 5b: Mixed Data with Categorical Features${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

if [ ! -f "$TRAINING_CSV" ] || [ ! -f "$TEST_CSV" ]; then
    echo -e "${RED}Skipping: CSV files not found${NC}"
else
    echo -e "${YELLOW}Training with mixed data (numeric + categorical)...${NC}"
    echo "   Features: response_time_ms (numeric), bytes_transferred (numeric),"
    echo "             endpoint (label), method (label), user_agent (hash)"
    echo ""

    # Define schema for mixed data encoding
    SCHEMA='{"response_time_ms": "numeric", "bytes_transferred": "numeric", "requests_in_session": "numeric", "endpoint": "label", "method": "label", "user_agent": "hash"}'

    # Extract training data as dict objects (not arrays)
    TRAINING_DICT_DATA=$(python3 -c "
import csv
import json
data = []
with open(\"$TRAINING_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'response_time_ms': int(row['response_time_ms']),
            'bytes_transferred': int(row['bytes_transferred']),
            'requests_in_session': int(row['requests_in_session']),
            'endpoint': row['endpoint'],
            'method': row['method'],
            'user_agent': row['user_agent']
        })
print(json.dumps(data))
" 2>/dev/null)

    # Train the model with mixed data
    FIT_MIXED_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"model\": \"api_log_detector_mixed\",
            \"backend\": \"isolation_forest\",
            \"data\": ${TRAINING_DICT_DATA},
            \"schema\": ${SCHEMA},
            \"contamination\": 0.01
        }")

    echo "$FIT_MIXED_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", \"?\")} samples in {data.get(\"training_time_ms\", 0):.1f}ms')
    encoder = data.get('encoder', {})
    if encoder:
        print(f'  Encoder schema: {encoder.get(\"schema\", {})}')
        print(f'  Features: {encoder.get(\"features\", [])}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    echo ""

    echo -e "${YELLOW}Detecting anomalies with mixed data...${NC}"
    echo "   The model can now detect anomalies based on BOTH numeric patterns"
    echo "   AND categorical features (like unusual user agents or endpoints)"
    echo ""

    # Extract test data as dict objects
    TEST_DICT_DATA=$(python3 -c "
import csv
import json
data = []
with open(\"$TEST_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'response_time_ms': int(row['response_time_ms']),
            'bytes_transferred': int(row['bytes_transferred']),
            'requests_in_session': int(row['requests_in_session']),
            'endpoint': row['endpoint'],
            'method': row['method'],
            'user_agent': row['user_agent']
        })
print(json.dumps(data))
" 2>/dev/null)

    # Detect anomalies (no schema needed - uses encoder from fit)
    DETECT_MIXED_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"model\": \"api_log_detector_mixed\",
            \"backend\": \"isolation_forest\",
            \"data\": ${TEST_DICT_DATA},
            \"threshold\": 0.75
        }")

    echo "Mixed Data Analysis Results:"
    python3 -c "
import sys, json, csv

# Load test CSV for context
test_rows = []
with open(\"$TEST_CSV\", 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_rows.append(row)

# Parse detection response
response = json.loads('''${DETECT_MIXED_RESPONSE}''')
anomaly_indices = {a['index'] for a in response.get('data', [])}
anomaly_scores = {a['index']: a['score'] for a in response.get('data', [])}

print('  ' + '-' * 120)
print(f'  {\"#\":<4} {\"Time(ms)\":<10} {\"Bytes\":<10} {\"Method\":<8} {\"Status\":<18} UserAgent (truncated)')
print('  ' + '-' * 120)

# Only show first 10 and any anomalies
shown = 0
for i, row in enumerate(test_rows):
    is_anomaly = i in anomaly_indices
    if shown < 8 or is_anomaly or i >= len(test_rows) - 3:
        status = f'🚨 ANOMALY ({anomaly_scores.get(i, 0):.2f})' if is_anomaly else '✓ Normal'
        time_ms = row['response_time_ms']
        bytes_t = row['bytes_transferred']
        method = row['method']
        user_agent = row['user_agent'][:50]
        print(f'  {i:<4} {time_ms:<10} {bytes_t:<10} {method:<8} {status:<18} {user_agent}')
        shown += 1
    elif shown == 8:
        print(f'  ... ({len(test_rows) - 11} more entries) ...')
        shown += 1

print('  ' + '-' * 120)
summary = response.get('summary', {})
print(f'  Total anomalies detected: {summary.get(\"anomalies_detected\", len(anomaly_indices))} out of {len(test_rows)} requests')

# Compare with numeric-only model
print('')
print('  Note: Mixed data model can detect anomalies based on categorical patterns')
print('        (e.g., unusual user agents, rare endpoints) in addition to numeric features.')
" 2>/dev/null

fi
echo ""

# ============================================================================
# Test 6: Production Workflow - Load Saved Model
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 6: Load Saved Model (Production)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Loading previously saved model...${NC}"
echo "   This simulates loading after server restart"
echo ""

LOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/load" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "sensor_anomaly_detector",
        "backend": "isolation_forest"
    }')

echo "Load Response:"
echo "$LOAD_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$LOAD_RESPONSE"
echo ""

if echo "$LOAD_RESPONSE" | grep -q '"loaded"'; then
    echo -e "${GREEN}✓ Model loaded and ready for inference!${NC}"

    # Test with loaded model
    echo ""
    echo -e "${YELLOW}Testing loaded model with new data...${NC}"

    TEST_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d '{
            "model": "sensor_anomaly_detector",
            "backend": "isolation_forest",
            "data": [[22.5], [100.0], [23.0]],
            "threshold": 0.5
        }')

    echo "$TEST_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    anomalies = data.get('data', [])
    print(f'  Detected {len(anomalies)} anomalies in test data')
    for a in anomalies:
        print(f'    - Index {a[\"index\"]}: score {a[\"score\"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${YELLOW}Load may have failed (this is expected on first run)${NC}"
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Production Workflow Summary:"
echo "  1. POST /v1/anomaly/fit     - Train model on normal data"
echo "  2. POST /v1/anomaly/save    - Save model to disk"
echo "  3. GET  /v1/anomaly/models  - List saved models"
echo "  4. POST /v1/anomaly/load    - Load model (after restart)"
echo "  5. POST /v1/anomaly/detect  - Detect anomalies"
echo ""
