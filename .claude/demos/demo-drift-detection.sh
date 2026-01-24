#!/bin/bash
# Demo: Alibi Detect Drift Detection
# =================================
# This demo shows how to use the drift detection endpoints to
# monitor data distribution changes.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration - Use LlamaFarm API (port 8005) for production usage
# Falls back to Universal Runtime (port 11545) for local testing
PORT="${LLAMAFARM_PORT:-${UNIVERSAL_PORT:-8005}}"
BASE_URL="${LLAMAFARM_URL:-http://localhost:${PORT}}"

# Helper function for generating Gaussian data using Box-Muller
generate_gaussian() {
    python3 -c "
import json
import random
import math
random.seed($1)
mean = $2
std = $3
n_samples = $4
n_features = $5

def gauss(mean, std):
    u1 = random.random()
    u2 = random.random()
    return mean + std * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

data = [[gauss(mean, std) for _ in range(n_features)] for _ in range(n_samples)]
print(json.dumps(data))
"
}

# Helper function for generating categorical data
generate_categorical() {
    python3 -c "
import json
import random
random.seed($1)
n_samples = $2
n_features = $3
n_categories = $4
# Optional bias - if > 0, bias towards category 0
bias = $5

data = []
for _ in range(n_samples):
    row = []
    for _ in range(n_features):
        if bias > 0 and random.random() < bias:
            row.append(0.0)
        else:
            row.append(float(random.randint(0, n_categories - 1)))
    data.append(row)
print(json.dumps(data))
"
}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              ALIBI DETECT DRIFT DETECTION DEMO                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Wait for server
echo -e "${YELLOW}⏳ Waiting for Universal Runtime on port ${PORT}...${NC}"
for i in {1..30}; do
    if curl -s "${BASE_URL}/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Server not responding after 30 seconds${NC}"
        exit 1
    fi
    sleep 1
done
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: List Available Drift Detectors
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: List Available Drift Detectors${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

DETECTORS_RESPONSE=$(curl -s "${BASE_URL}/v1/drift/detectors")
echo "Available detectors:"
echo "${DETECTORS_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data['detectors']:
    multi = '(multivariate)' if d['multivariate'] else '(univariate)'
    print(f\"  • {d['name']}: {d['description']} {multi}\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Fit a KS Drift Detector on Reference Data
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Fit KS Drift Detector on Reference Data${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate reference data: 100 samples of 3 features from N(0, 1)
echo "Generating reference data: 100 samples × 3 features from N(0, 1)"
REFERENCE_DATA=$(generate_gaussian 42 0 1 100 3)

echo "Fitting KS drift detector..."
FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-ks\",
        \"detector\": \"ks\",
        \"reference_data\": ${REFERENCE_DATA},
        \"feature_names\": [\"feature_a\", \"feature_b\", \"feature_c\"],
        \"description\": \"Demo KS drift detector for monitoring\"
    }")

echo "${FIT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Model: {data['model']}\")
print(f\"  Detector: {data['detector']}\")
print(f\"  Reference Size: {data['reference_size']}\")
print(f\"  Features: {data['n_features']}\")
print(f\"  Training Time: {data['training_time_ms']:.2f}ms\")
"
echo -e "${GREEN}✓ Drift detector fitted and saved${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Test with Data from Same Distribution (No Drift)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Test with Same Distribution (Expect No Drift)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate test data from same distribution
echo "Generating test data: 50 samples from N(0, 1)"
SAME_DIST_DATA=$(generate_gaussian 123 0 1 50 3)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-ks\",
        \"data\": ${SAME_DIST_DATA}
    }")

echo "${DETECT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
result = data['result']
drift_status = '🔴 DRIFT DETECTED' if result['is_drift'] else '🟢 NO DRIFT'
print(f\"  Result: {drift_status}\")
print(f\"  P-value: {result['p_value']:.4f}\")
print(f\"  Threshold: {result['threshold']}\")
print(f\"  Detection Time: {data['detection_time_ms']:.2f}ms\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Test with Shifted Distribution (Drift Expected)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: Test with Shifted Distribution (Expect Drift)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate test data from shifted distribution (mean = 3)
echo "Generating test data: 50 samples from N(3, 1) - SHIFTED MEAN"
SHIFTED_DATA=$(generate_gaussian 456 3 1 50 3)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-ks\",
        \"data\": ${SHIFTED_DATA}
    }")

echo "${DETECT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
result = data['result']
drift_status = '🔴 DRIFT DETECTED' if result['is_drift'] else '🟢 NO DRIFT'
print(f\"  Result: {drift_status}\")
print(f\"  P-value: {result['p_value']:.6f}\")
print(f\"  Threshold: {result['threshold']}\")
print(f\"  Detection Time: {data['detection_time_ms']:.2f}ms\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Test with Variance Change (Different Type of Drift)
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Test with Variance Change${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate test data with increased variance (std = 3)
echo "Generating test data: 50 samples from N(0, 3) - INCREASED VARIANCE"
VARIANCE_DATA=$(generate_gaussian 789 0 3 50 3)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-ks\",
        \"data\": ${VARIANCE_DATA}
    }")

echo "${DETECT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
result = data['result']
drift_status = '🔴 DRIFT DETECTED' if result['is_drift'] else '🟢 NO DRIFT'
print(f\"  Result: {drift_status}\")
print(f\"  P-value: {result['p_value']:.6f}\")
print(f\"  Threshold: {result['threshold']}\")
print(f\"  Detection Time: {data['detection_time_ms']:.2f}ms\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Check Drift Detector Status
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6: Check Drift Detector Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

STATUS_RESPONSE=$(curl -s "${BASE_URL}/v1/drift/status/demo-drift-ks")
echo "${STATUS_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Model: {data['model']}\")
print(f\"  Detector: {data['detector']}\")
print(f\"  Is Fitted: {data['is_fitted']}\")
print(f\"  Reference Size: {data['reference_size']}\")
print(f\"  Detection Count: {data['detection_count']}\")
print(f\"  Drift Count: {data['drift_count']}\")
if data.get('last_detection'):
    ld = data['last_detection']
    print(f\"  Last Detection: drift={ld['is_drift']}, p={ld['p_value']:.4f}\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Fit Chi-Squared Detector for Categorical Data
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 7: Fit Chi-Squared Detector for Categorical Data${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Generate categorical reference data (0-4 categories)
echo "Generating categorical reference data: 100 samples × 2 features (0-4 categories)"
CATEGORICAL_REF=$(generate_categorical 42 100 2 5 0)

echo "Fitting Chi-squared drift detector..."
FIT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-chi2\",
        \"detector\": \"chi2\",
        \"reference_data\": ${CATEGORICAL_REF},
        \"feature_names\": [\"category_a\", \"category_b\"]
    }")

echo "${FIT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Model: {data['model']}\")
print(f\"  Detector: {data['detector']}\")
print(f\"  Reference Size: {data['reference_size']}\")
"
echo -e "${GREEN}✓ Chi-squared detector fitted${NC}"
echo ""

# Test with same categorical distribution
echo "Testing with same categorical distribution..."
SAME_CAT=$(generate_categorical 123 50 2 5 0)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-chi2\",
        \"data\": ${SAME_CAT}
    }")

echo "${DETECT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
result = data['result']
drift_status = '🔴 DRIFT DETECTED' if result['is_drift'] else '🟢 NO DRIFT'
print(f\"  Result: {drift_status}\")
print(f\"  P-value: {result['p_value']:.4f}\")
"
echo ""

# Test with different categorical distribution (biased towards category 0)
echo "Testing with biased categorical distribution (70% category 0)..."
BIASED_CAT=$(generate_categorical 456 50 2 5 0.7)

DETECT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/drift/detect" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-drift-chi2\",
        \"data\": ${BIASED_CAT}
    }")

echo "${DETECT_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
result = data['result']
drift_status = '🔴 DRIFT DETECTED' if result['is_drift'] else '🟢 NO DRIFT'
print(f\"  Result: {drift_status}\")
print(f\"  P-value: {result['p_value']:.4f}\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 8: List All Saved Models
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 8: List All Saved Drift Models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

MODELS_RESPONSE=$(curl -s "${BASE_URL}/v1/drift/models")
echo "${MODELS_RESPONSE}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Found {len(data['models'])} saved model(s):\")
for m in data['models']:
    print(f\"  • {m['name']} ({m['detector']}): {m.get('description', 'No description')}\")
"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Clean Up Demo Models
# ═══════════════════════════════════════════════════════════════════════════
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 9: Clean Up Demo Models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

curl -s -X DELETE "${BASE_URL}/v1/drift/models/demo-drift-ks" | python3 -c "
import json, sys
data = json.load(sys.stdin)
status = '✓' if data['deleted'] else '⚠'
print(f\"  {status} demo-drift-ks: deleted={data['deleted']}\")
"

curl -s -X DELETE "${BASE_URL}/v1/drift/models/demo-drift-chi2" | python3 -c "
import json, sys
data = json.load(sys.stdin)
status = '✓' if data['deleted'] else '⚠'
print(f\"  {status} demo-drift-chi2: deleted={data['deleted']}\")
"
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    DRIFT DETECTION DEMO COMPLETE               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Key Takeaways:"
echo "  • KS (Kolmogorov-Smirnov): Univariate drift detection for continuous data"
echo "  • Chi-squared: Drift detection for categorical data"
echo "  • Low p-values indicate significant drift from reference distribution"
echo "  • Monitor drift_count/detection_count ratio for ongoing drift rate"
echo ""
echo "Use Cases:"
echo "  • ML model monitoring: Detect input feature drift"
echo "  • Data quality: Identify distribution changes in pipelines"
echo "  • A/B testing: Compare feature distributions across cohorts"
echo ""
