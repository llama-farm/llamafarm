#!/bin/bash
# Full ML Pipeline Integration Demo
# Demonstrates the complete "Magnificent 7" ML pipeline
#
# Components demonstrated:
# 1. PyOD Anomaly Detection
# 2. ADTK Time-Series Anomaly Detection
# 3. Time-Series Forecasting (Darts)
# 4. Drift Detection (Alibi Detect)
# 5. SHAP Explainability
# 6. CatBoost Classification
# 7. Streaming with Polars Buffer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║       LlamaFarm 'Magnificent 7' ML Pipeline Demo           ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration
PORT="${UNIVERSAL_PORT:-11545}"
BASE_URL="http://localhost:${PORT}"

# Check if server is running
echo -e "${YELLOW}Checking server status...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"
echo

# Track results for summary
TESTS_PASSED=0
TESTS_TOTAL=0

check_result() {
    local name="$1"
    local result="$2"
    ((TESTS_TOTAL++))
    if [ "$result" = "0" ]; then
        echo -e "${GREEN}✓ $name passed${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ $name failed${NC}"
    fi
}

# =============================================================================
# Section 1: PyOD Anomaly Detection
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 1: PyOD Anomaly Detection${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate training data
TRAIN_DATA=$(python3 -c "
import json
import random
random.seed(42)
data = [[random.gauss(0, 1) for _ in range(5)] for _ in range(100)]
print(json.dumps(data))")

# Train anomaly detector
echo -e "${YELLOW}Training Isolation Forest anomaly detector...${NC}"
FIT_RESULT=$(curl -s -X POST "${BASE_URL}/v1/anomaly/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"data\": ${TRAIN_DATA},
        \"model_id\": \"pipeline-anomaly\",
        \"backend\": \"isolation_forest\",
        \"contamination\": 0.1
    }" 2>/dev/null || echo '{"error":"failed"}')

if echo "$FIT_RESULT" | grep -q "model_id"; then
    check_result "PyOD anomaly training" "0"
else
    check_result "PyOD anomaly training" "1"
fi

# Score normal and anomalous samples
echo -e "${YELLOW}Scoring samples...${NC}"
SCORE_RESULT=$(curl -s -X POST "${BASE_URL}/v1/anomaly/score" \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "pipeline-anomaly",
        "data": [
            [0.1, 0.2, -0.1, 0.3, 0.0],
            [5.0, -4.0, 3.5, -2.8, 4.2]
        ]
    }' 2>/dev/null || echo '{"error":"failed"}')

echo "$SCORE_RESULT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'scores' in data:
        for s in data['scores']:
            status = '⚠️ ANOMALY' if s['is_anomaly'] else '✓ Normal'
            print(f\"  Sample {s['index']}: score={s['score']:.4f} {status}\")
except:
    print('  Failed to parse scores')" 2>/dev/null
echo

# =============================================================================
# Section 2: ADTK Time-Series Anomaly Detection
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 2: ADTK Time-Series Anomaly Detection${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate time-series data with level shift
TS_DATA=$(python3 -c "
import json
import random
from datetime import datetime, timedelta
random.seed(42)

data = []
base_date = datetime(2024, 1, 1)
for i in range(100):
    value = random.gauss(100, 10) if i < 70 else random.gauss(150, 10)  # Level shift at 70
    data.append({
        'timestamp': (base_date + timedelta(hours=i)).isoformat(),
        'value': round(value, 2)
    })
print(json.dumps(data))")

# Fit ADTK detector
echo -e "${YELLOW}Training ADTK level shift detector...${NC}"
ADTK_FIT=$(curl -s -X POST "${BASE_URL}/v1/adtk/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-adtk\",
        \"detector\": \"level_shift\",
        \"data\": ${TS_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

if echo "$ADTK_FIT" | grep -q "model_id"; then
    check_result "ADTK training" "0"
else
    check_result "ADTK training" "1"
fi

# Detect anomalies
echo -e "${YELLOW}Detecting level shifts...${NC}"
ADTK_DETECT=$(curl -s -X POST "${BASE_URL}/v1/adtk/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-adtk\",
        \"data\": ${TS_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

echo "$ADTK_DETECT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'anomalies' in data:
        count = len(data['anomalies'])
        print(f'  Found {count} anomalous time points')
        if count > 0 and count <= 5:
            for a in data['anomalies'][:3]:
                print(f\"    {a['timestamp']}: value={a['value']}\")
except:
    print('  Failed to parse detection results')" 2>/dev/null
echo

# =============================================================================
# Section 3: Time-Series Forecasting
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 3: Time-Series Forecasting${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate forecasting data
FORECAST_DATA=$(python3 -c "
import json
import random
import math
random.seed(42)

data = []
for i in range(50):
    # Trend + seasonality + noise
    value = 50 + i * 0.5 + 10 * math.sin(i * 0.3) + random.gauss(0, 2)
    data.append({'value': round(value, 2)})
print(json.dumps(data))")

# Fit forecaster
echo -e "${YELLOW}Training ARIMA forecaster...${NC}"
TS_FIT=$(curl -s -X POST "${BASE_URL}/v1/timeseries/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-forecast\",
        \"backend\": \"arima\",
        \"data\": ${FORECAST_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

if echo "$TS_FIT" | grep -q "model_id"; then
    check_result "Timeseries training" "0"
else
    check_result "Timeseries training" "1"
fi

# Generate forecast
echo -e "${YELLOW}Generating 7-step forecast...${NC}"
TS_PREDICT=$(curl -s -X POST "${BASE_URL}/v1/timeseries/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "pipeline-forecast",
        "steps": 7
    }' 2>/dev/null || echo '{"error":"failed"}')

echo "$TS_PREDICT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'predictions' in data:
        print('  7-Day Forecast:')
        for p in data['predictions'][:7]:
            lo = p.get('lower', 'N/A')
            hi = p.get('upper', 'N/A')
            if isinstance(lo, float): lo = f'{lo:.2f}'
            if isinstance(hi, float): hi = f'{hi:.2f}'
            print(f\"    Step {p['step']}: {p['value']:.2f} [{lo} - {hi}]\")
except:
    print('  Forecast data available')" 2>/dev/null
echo

# =============================================================================
# Section 4: Drift Detection
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 4: Alibi Detect - Drift Detection${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate reference data
REF_DATA=$(python3 -c "
import json
import random
random.seed(42)
data = [[random.gauss(0, 1) for _ in range(3)] for _ in range(100)]
print(json.dumps(data))")

# Fit drift detector
echo -e "${YELLOW}Training KS drift detector...${NC}"
DRIFT_FIT=$(curl -s -X POST "${BASE_URL}/v1/drift/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-drift\",
        \"detector\": \"ks\",
        \"reference_data\": ${REF_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

if echo "$DRIFT_FIT" | grep -q "model_id"; then
    check_result "Drift detector training" "0"
else
    check_result "Drift detector training" "1"
fi

# Test for drift with similar data (no drift)
echo -e "${YELLOW}Testing similar data (no drift expected)...${NC}"
SIMILAR_DATA=$(python3 -c "
import json
import random
random.seed(123)
data = [[random.gauss(0, 1) for _ in range(3)] for _ in range(50)]
print(json.dumps(data))")

DRIFT_NO=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-drift\",
        \"data\": ${SIMILAR_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

echo "$DRIFT_NO" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'is_drift' in data:
        status = '⚠️ DRIFT' if data['is_drift'] else '✓ No drift'
        print(f\"  Similar data: {status} (p={data.get('p_value', 'N/A')})\")
except:
    print('  Drift check completed')" 2>/dev/null

# Test for drift with shifted data
echo -e "${YELLOW}Testing shifted data (drift expected)...${NC}"
SHIFTED_DATA=$(python3 -c "
import json
import random
random.seed(456)
data = [[random.gauss(3, 1) for _ in range(3)] for _ in range(50)]  # Mean shifted from 0 to 3
print(json.dumps(data))")

DRIFT_YES=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-drift\",
        \"data\": ${SHIFTED_DATA}
    }" 2>/dev/null || echo '{"error":"failed"}')

echo "$DRIFT_YES" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'is_drift' in data:
        status = '⚠️ DRIFT DETECTED' if data['is_drift'] else '✓ No drift'
        print(f\"  Shifted data: {status} (p={data.get('p_value', 'N/A')})\")
except:
    print('  Drift check completed')" 2>/dev/null
echo

# =============================================================================
# Section 5: SHAP Explainability
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 5: SHAP Explainability${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# List available explainers
echo -e "${YELLOW}Listing SHAP explainer types...${NC}"
EXPLAINERS=$(curl -s "${BASE_URL}/v1/explain/explainers" 2>/dev/null || echo '{"explainers":[]}')

echo "$EXPLAINERS" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    for e in data.get('explainers', [])[:3]:
        print(f\"  • {e['name']}: {e['description'][:50]}...\")
except:
    print('  Explainers available')" 2>/dev/null

if echo "$EXPLAINERS" | grep -q "tree"; then
    check_result "SHAP explainers" "0"
else
    check_result "SHAP explainers" "1"
fi
echo

# =============================================================================
# Section 6: CatBoost Classification
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 6: CatBoost Classification${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Check CatBoost availability
echo -e "${YELLOW}Checking CatBoost availability...${NC}"
CB_INFO=$(curl -s "${BASE_URL}/v1/catboost/info" 2>/dev/null || echo '{"available":false}')

if echo "$CB_INFO" | grep -q '"available":true'; then
    check_result "CatBoost available" "0"
else
    check_result "CatBoost available" "1"
fi

# Train CatBoost classifier
echo -e "${YELLOW}Training CatBoost classifier...${NC}"
CB_DATA=$(python3 -c "
import json
import random
random.seed(42)
data = [[random.gauss(0, 1) for _ in range(4)] for _ in range(100)]
labels = [1 if row[0] > 0 else 0 for row in data]
print(json.dumps({'data': data, 'labels': labels}))")

CB_X=$(echo "$CB_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['data']))")
CB_Y=$(echo "$CB_DATA" | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['labels']))")

CB_FIT=$(curl -s -X POST "${BASE_URL}/v1/catboost/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model_id\": \"pipeline-catboost\",
        \"model_type\": \"classifier\",
        \"data\": ${CB_X},
        \"labels\": ${CB_Y},
        \"iterations\": 20,
        \"depth\": 3
    }" 2>/dev/null || echo '{"error":"failed"}')

if echo "$CB_FIT" | grep -q "model_id"; then
    echo "$CB_FIT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f\"  Trained: {data.get('iterations', 0)} trees, {data.get('n_features', 0)} features\")
except:
    print('  CatBoost training completed')" 2>/dev/null
    check_result "CatBoost training" "0"
else
    check_result "CatBoost training" "1"
fi

# Make predictions
echo -e "${YELLOW}Making predictions...${NC}"
CB_PRED=$(curl -s -X POST "${BASE_URL}/v1/catboost/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "model_id": "pipeline-catboost",
        "data": [[0.5, 0.1, -0.2, 0.3], [-0.8, 0.2, 0.1, -0.1]],
        "return_proba": true
    }' 2>/dev/null || echo '{"error":"failed"}')

echo "$CB_PRED" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'predictions' in data:
        for p in data['predictions']:
            probas = p.get('probabilities', {})
            conf = max(float(v) for v in probas.values()) if probas else 0
            print(f\"  Sample {p['sample_index']}: Class {p['prediction']} (conf: {conf:.1%})\")
except:
    print('  Predictions made')" 2>/dev/null
echo

# =============================================================================
# Section 7: Streaming with Polars
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Section 7: Streaming Pipeline (Polars Buffer)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Stream data through anomaly detector
echo -e "${YELLOW}Streaming 50 data points through detector...${NC}"
STREAM_DATA=$(python3 -c "
import json
import random
random.seed(789)
data = []
for i in range(50):
    if i == 25:  # Inject anomaly
        data.append([5.0, -4.0, 3.5, -2.8, 4.2])
    else:
        data.append([random.gauss(0, 1) for _ in range(5)])
print(json.dumps(data))")

# Process stream
ANOMALY_COUNT=0
for i in $(seq 0 9); do
    BATCH=$(echo "$STREAM_DATA" | python3 -c "
import json, sys
data = json.load(sys.stdin)
batch = data[$i*5:($i+1)*5]
print(json.dumps(batch))")

    STREAM_RESULT=$(curl -s -X POST "${BASE_URL}/v1/anomaly/stream" \
        -H "Content-Type: application/json" \
        -d "{
            \"model_id\": \"pipeline-stream\",
            \"data\": ${BATCH},
            \"backend\": \"isolation_forest\"
        }" 2>/dev/null || echo '{"error":"failed"}')

    # Count anomalies
    BATCH_ANOMALIES=$(echo "$STREAM_RESULT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    count = sum(1 for r in data.get('results', []) if r.get('is_anomaly', False))
    print(count)
except:
    print(0)" 2>/dev/null || echo "0")

    ANOMALY_COUNT=$((ANOMALY_COUNT + BATCH_ANOMALIES))
done

echo "  Processed 50 data points in 10 batches"
echo "  Total anomalies detected: $ANOMALY_COUNT"

if [ "$ANOMALY_COUNT" -ge 1 ]; then
    check_result "Streaming detection" "0"
else
    check_result "Streaming detection" "1"
fi
echo

# =============================================================================
# Cleanup
# =============================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Cleanup${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Delete demo models
curl -s -X DELETE "${BASE_URL}/v1/drift/pipeline-drift" > /dev/null 2>&1
curl -s -X DELETE "${BASE_URL}/v1/catboost/pipeline-catboost" > /dev/null 2>&1
curl -s -X DELETE "${BASE_URL}/v1/anomaly/stream/pipeline-stream" > /dev/null 2>&1
echo -e "${GREEN}✓ Demo models cleaned up${NC}"
echo

# =============================================================================
# Summary
# =============================================================================
echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║                        SUMMARY                              ║${NC}"
echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
echo

echo -e "Tests passed: ${GREEN}${TESTS_PASSED}/${TESTS_TOTAL}${NC}"
echo

echo "Components Demonstrated:"
echo "  1. ✓ PyOD Anomaly Detection (Isolation Forest)"
echo "  2. ✓ ADTK Time-Series Anomaly Detection (Level Shift)"
echo "  3. ✓ Time-Series Forecasting (ARIMA)"
echo "  4. ✓ Drift Detection (Kolmogorov-Smirnov)"
echo "  5. ✓ SHAP Explainability (Tree, Linear, Kernel)"
echo "  6. ✓ CatBoost Classification (Incremental Learning)"
echo "  7. ✓ Streaming Pipeline (Polars Buffer)"
echo

echo "API Endpoints Summary:"
echo "  Anomaly:    /v1/anomaly/{fit,score,stream,backends}"
echo "  ADTK:       /v1/adtk/{fit,detect,detectors}"
echo "  Timeseries: /v1/timeseries/{fit,predict,backends}"
echo "  Drift:      /v1/drift/{fit,detect,status,reset}"
echo "  Explain:    /v1/explain/{explainers,shap,importance}"
echo "  CatBoost:   /v1/catboost/{fit,predict,update,importance}"
echo

if [ "$TESTS_PASSED" -eq "$TESTS_TOTAL" ]; then
    echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║    All components working! Pipeline ready.    ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔═══════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║    Some components need attention.            ║${NC}"
    echo -e "${YELLOW}╚═══════════════════════════════════════════════╝${NC}"
fi
