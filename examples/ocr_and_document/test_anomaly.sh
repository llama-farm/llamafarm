#!/bin/bash
# Test Anomaly Detection endpoint
#
# This script demonstrates:
# 1. Fitting an anomaly detector on normal data
# 2. Testing all three normalization methods (standardization, zscore, raw)
# 3. Comparing results across methods
# 4. Saving and loading trained models
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
CYAN='\033[0;36m'
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
# Test 1: Standardization Method (Default - 0-1 range)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Standardization Method (Default)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training with standardization normalization...${NC}"
echo "   Method: Sigmoid transformation to 0-1 range"
echo "   Threshold: 0.5 (default)"
echo "   Simulating: Temperature readings (normal range: 20-25°C)"
echo ""

# Generate normal data (temperatures between 20-25)
FIT_STD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_standardization",
        "backend": "isolation_forest",
        "normalization": "standardization",
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

echo "$FIT_STD_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", \"?\")} samples')
    print(f'  Threshold: {data.get(\"model_params\", {}).get(\"threshold\", \"N/A\"):.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Test with anomalies
echo -e "${YELLOW}Detecting anomalies (standardization)...${NC}"
DETECT_STD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_standardization",
        "backend": "isolation_forest",
        "normalization": "standardization",
        "data": [
            [22.0], [23.5], [5.0], [21.5], [50.0],
            [24.0], [-10.0], [22.8], [35.0], [23.2]
        ],
        "threshold": 0.5
    }')

echo "Test data: [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]"
echo ""
echo "$DETECT_STD_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    readings = [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]
    anomalies = data.get('data', [])
    summary = data.get('summary', {})
    print(f'  Anomalies found: {summary.get(\"anomalies_detected\", len(anomalies))}')
    print(f'  Threshold: {summary.get(\"threshold\", \"N/A\")}')
    if anomalies:
        print(f'  Anomalous readings (score 0-1, higher = more anomalous):')
        for a in anomalies:
            idx = a['index']
            print(f'    - {readings[idx]}°C: score={a[\"score\"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Z-Score Method (Standard Deviations)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Z-Score Method (Std Deviations)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training with zscore normalization...${NC}"
echo "   Method: (score - mean) / std"
echo "   Threshold: 2.0 (default, meaning 2 std devs)"
echo "   Interpretation: 2.0 = unusual, 3.0 = rare, 4.0+ = extreme"
echo ""

FIT_ZSCORE_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_zscore",
        "backend": "isolation_forest",
        "normalization": "zscore",
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

echo "$FIT_ZSCORE_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", \"?\")} samples')
    print(f'  Threshold: {data.get(\"model_params\", {}).get(\"threshold\", \"N/A\"):.4f} std devs')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Test with anomalies
echo -e "${YELLOW}Detecting anomalies (zscore)...${NC}"
DETECT_ZSCORE_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_zscore",
        "backend": "isolation_forest",
        "normalization": "zscore",
        "data": [
            [22.0], [23.5], [5.0], [21.5], [50.0],
            [24.0], [-10.0], [22.8], [35.0], [23.2]
        ],
        "threshold": 2.0
    }')

echo "Test data: [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]"
echo ""
echo "$DETECT_ZSCORE_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    readings = [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]
    anomalies = data.get('data', [])
    summary = data.get('summary', {})
    print(f'  Anomalies found: {summary.get(\"anomalies_detected\", len(anomalies))}')
    print(f'  Threshold: {summary.get(\"threshold\", \"N/A\")} std devs')
    if anomalies:
        print(f'  Anomalous readings (score = std deviations from normal):')
        for a in anomalies:
            idx = a['index']
            score = a['score']
            severity = 'unusual' if score < 3 else 'rare' if score < 4 else 'EXTREME'
            print(f'    - {readings[idx]}°C: {score:.2f} std devs ({severity})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 3: Raw Score Method (Backend Native)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Raw Score Method (Backend Native)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training with raw normalization...${NC}"
echo "   Method: No normalization, backend-native scores"
echo "   Isolation Forest range: ~-0.5 to 0.5"
echo "   Higher = more anomalous (after negation)"
echo ""

FIT_RAW_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_raw",
        "backend": "isolation_forest",
        "normalization": "raw",
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

echo "$FIT_RAW_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  ✓ Trained on {data.get(\"samples_fitted\", \"?\")} samples')
    threshold = data.get('model_params', {}).get('threshold', 'N/A')
    print(f'  Threshold: {threshold:.4f} (raw isolation forest score)')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Test with anomalies - use a higher threshold for raw scores
echo -e "${YELLOW}Detecting anomalies (raw)...${NC}"
DETECT_RAW_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/detect" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_raw",
        "backend": "isolation_forest",
        "normalization": "raw",
        "data": [
            [22.0], [23.5], [5.0], [21.5], [50.0],
            [24.0], [-10.0], [22.8], [35.0], [23.2]
        ],
        "threshold": 0.1
    }')

echo "Test data: [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]"
echo ""
echo "$DETECT_RAW_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    readings = [22.0, 23.5, 5.0, 21.5, 50.0, 24.0, -10.0, 22.8, 35.0, 23.2]
    anomalies = data.get('data', [])
    summary = data.get('summary', {})
    print(f'  Anomalies found: {summary.get(\"anomalies_detected\", len(anomalies))}')
    print(f'  Threshold: {summary.get(\"threshold\", \"N/A\")} (raw score)')
    if anomalies:
        print(f'  Anomalous readings (raw isolation forest scores):')
        for a in anomalies:
            idx = a['index']
            print(f'    - {readings[idx]}°C: raw_score={a[\"raw_score\"]:.4f}, score={a[\"score\"]:.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 4: Compare All Methods Side-by-Side
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 4: Compare All Methods Side-by-Side${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Scoring same data with all three methods...${NC}"
echo ""

# Score (not detect) to see all scores
SCORE_STD=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_standardization",
        "backend": "isolation_forest",
        "normalization": "standardization",
        "data": [[22.0], [5.0], [50.0], [-10.0]]
    }')

SCORE_ZSCORE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_zscore",
        "backend": "isolation_forest",
        "normalization": "zscore",
        "data": [[22.0], [5.0], [50.0], [-10.0]]
    }')

SCORE_RAW=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_raw",
        "backend": "isolation_forest",
        "normalization": "raw",
        "data": [[22.0], [5.0], [50.0], [-10.0]]
    }')

python3 -c "
import json

readings = [22.0, 5.0, 50.0, -10.0]
labels = ['Normal (22°C)', 'Cold (5°C)', 'Hot (50°C)', 'Freezing (-10°C)']

std_data = json.loads('''${SCORE_STD}''').get('data', [])
zscore_data = json.loads('''${SCORE_ZSCORE}''').get('data', [])
raw_data = json.loads('''${SCORE_RAW}''').get('data', [])

print('  ' + '-' * 85)
print(f'  {\"Reading\":<20} {\"Standardization\":<20} {\"Z-Score\":<20} {\"Raw Score\":<20}')
print(f'  {\"\":<20} {\"(0-1, >0.5=anom)\":<20} {\"(std devs, >2=anom)\":<20} {\"(IF native)\":<20}')
print('  ' + '-' * 85)

for i, label in enumerate(labels):
    std_score = std_data[i]['score'] if i < len(std_data) else 'N/A'
    z_score = zscore_data[i]['score'] if i < len(zscore_data) else 'N/A'
    raw_score = raw_data[i]['score'] if i < len(raw_data) else 'N/A'

    std_str = f'{std_score:.4f}' if isinstance(std_score, float) else std_score
    z_str = f'{z_score:.4f}' if isinstance(z_score, float) else z_score
    raw_str = f'{raw_score:.4f}' if isinstance(raw_score, float) else raw_score

    print(f'  {label:<20} {std_str:<20} {z_str:<20} {raw_str:<20}')

print('  ' + '-' * 85)
print('')
print('  Key observations:')
print('    - Standardization: Bounded 0-1, good for general use')
print('    - Z-Score: Unbounded, shows magnitude in std deviations')
print('    - Raw: Backend-native, useful for debugging')
" 2>/dev/null
echo ""

# ============================================================================
# Test 5: Different Backends with Z-Score
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 5: Different Backends (with Z-Score)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Training three different backends...${NC}"
echo ""

# Isolation Forest
curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_isolation_forest",
        "backend": "isolation_forest",
        "normalization": "zscore",
        "data": [
            [22.1], [23.5], [21.8], [24.2], [22.7],
            [23.1], [21.5], [24.8], [22.3], [23.9],
            [21.2], [24.5], [22.8], [23.2], [21.9]
        ],
        "contamination": 0.05
    }' > /dev/null

echo "  ✓ Isolation Forest trained"

# One Class SVM
curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_one_class_svm",
        "backend": "one_class_svm",
        "normalization": "zscore",
        "data": [
            [22.1], [23.5], [21.8], [24.2], [22.7],
            [23.1], [21.5], [24.8], [22.3], [23.9],
            [21.2], [24.5], [22.8], [23.2], [21.9]
        ],
        "contamination": 0.05
    }' > /dev/null

echo "  ✓ One Class SVM trained"

# Local Outlier Factor
curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_local_outlier_factor",
        "backend": "local_outlier_factor",
        "normalization": "zscore",
        "data": [
            [22.1], [23.5], [21.8], [24.2], [22.7],
            [23.1], [21.5], [24.8], [22.3], [23.9],
            [21.2], [24.5], [22.8], [23.2], [21.9]
        ],
        "contamination": 0.05
    }' > /dev/null

echo "  ✓ Local Outlier Factor trained"
echo ""

echo -e "${YELLOW}Comparing backend scores (zscore normalization)...${NC}"
echo ""

SCORE_IF=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_isolation_forest",
        "backend": "isolation_forest",
        "normalization": "zscore",
        "data": [[22.0], [5.0], [50.0]]
    }')

SCORE_SVM=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_one_class_svm",
        "backend": "one_class_svm",
        "normalization": "zscore",
        "data": [[22.0], [5.0], [50.0]]
    }')

SCORE_LOF=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "backend_local_outlier_factor",
        "backend": "local_outlier_factor",
        "normalization": "zscore",
        "data": [[22.0], [5.0], [50.0]]
    }')

python3 -c "
import json

readings = ['Normal (22°C)', 'Cold (5°C)', 'Hot (50°C)']

if_data = json.loads('''${SCORE_IF}''').get('data', [])
svm_data = json.loads('''${SCORE_SVM}''').get('data', [])
lof_data = json.loads('''${SCORE_LOF}''').get('data', [])

print('  ' + '-' * 80)
print(f'  {\"Reading\":<20} {\"Isolation Forest\":<20} {\"One Class SVM\":<20} {\"LOF\":<20}')
print('  ' + '-' * 80)

for i, label in enumerate(readings):
    if_score = f'{if_data[i][\"score\"]:.2f} std' if i < len(if_data) else 'N/A'
    svm_score = f'{svm_data[i][\"score\"]:.2f} std' if i < len(svm_data) else 'N/A'
    lof_score = f'{lof_data[i][\"score\"]:.2f} std' if i < len(lof_data) else 'N/A'

    print(f'  {label:<20} {if_score:<20} {svm_score:<20} {lof_score:<20}')

print('  ' + '-' * 80)
print('')
print('  Note: With zscore normalization, all backends use the same scale!')
print('        Scores > 2.0 std are considered anomalous.')
" 2>/dev/null
echo ""

# ============================================================================
# Test 6: Save and Load Model
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 6: Save and Load Model${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Note: Models now autosave during fit, no explicit save endpoint needed

echo -e "${YELLOW}Loading saved model...${NC}"
LOAD_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/anomaly/load" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d '{
        "model": "temp_zscore",
        "backend": "isolation_forest"
    }')

echo "$LOAD_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('loaded'):
        print(f'  ✓ Model loaded and ready')
        print(f'  Normalization: {data.get(\"normalization\", \"N/A\")}')
    else:
        print(f'  Load response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Normalization Methods Summary:"
echo ""
echo "  1. standardization (default)"
echo "     - Score range: 0-1"
echo "     - Default threshold: 0.5"
echo "     - Best for: General use, bounded scores"
echo ""
echo "  2. zscore"
echo "     - Score range: unbounded (standard deviations)"
echo "     - Default threshold: 2.0"
echo "     - Best for: Statistical interpretation"
echo "     - Interpretation: 2=unusual, 3=rare, 4+=extreme"
echo ""
echo "  3. raw"
echo "     - Score range: backend-specific"
echo "     - Default threshold: 0.0 (set your own!)"
echo "     - Best for: Debugging, advanced users"
echo ""
echo "API Endpoints:"
echo "  POST /v1/anomaly/fit     - Train model (autosaves, add 'normalization' param)"
echo "  POST /v1/anomaly/score   - Score all data points"
echo "  POST /v1/anomaly/detect  - Return only anomalies"
echo "  POST /v1/anomaly/load    - Load model from disk"
echo "  GET  /v1/anomaly/models  - List saved models"
echo ""
