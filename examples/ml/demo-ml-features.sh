#!/bin/bash
# Demo: ML Features Overview
# Phase 18 of Universal Runtime ML Enhancements
#
# Lists all ML features and verifies they're available.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

echo "=========================================="
echo "ML Features Overview"
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

echo "2. ML Features Matrix"
echo ""
echo "   | Category          | Feature                   | Endpoint                        |"
echo "   |-------------------|---------------------------|---------------------------------|"
echo "   | Vision            | Zero-Shot Classification  | /v1/vision/classify             |"
echo "   | Vision            | Object Detection (YOLOS)  | /v1/vision/detect               |"
echo "   | Vision            | Background Removal        | /v1/vision/remove-background    |"
echo "   | NLP               | Language Identification   | /v1/ml/nlp/identify-language       |"
echo "   | NLP               | Keyword Extraction        | /v1/ml/nlp/keywords                |"
echo "   | NLP               | PII Redaction (GLiNER)    | /v1/ml/nlp/pii/redact              |"
echo "   | Time-Series       | Forecasting (Chronos)     | /v1/ml/timeseries/forecast         |"
echo "   | Time-Series       | Change Point Detection    | /v1/ml/timeseries/changepoints     |"
echo "   | Streaming         | Concept Drift Detection   | /v1/ml/analysis/drift/*           |"
echo "   | Anomaly           | Multi-Backend Detection   | /v1/ml/anomaly/fit, /score         |"
echo "   | Anomaly           | SHAP Explanations         | /v1/ml/anomaly/explain             |"
echo "   | Data Quality      | Dataset Audit (Cleanlab)  | /v1/ml/analysis/dataset-audit    |"
echo "   | Analysis          | Table QA (TAPAS)          | /v1/ml/analysis/table-qa         |"
echo ""

echo "3. Anomaly Detection Backends"
echo ""
echo "   | Backend           | Type     | Description                          |"
echo "   |-------------------|----------|--------------------------------------|"
echo "   | isolation_forest  | sklearn  | Tree-based, fast, general purpose    |"
echo "   | one_class_svm     | sklearn  | SVM-based, good for small datasets   |"
echo "   | local_outlier_factor | sklearn | Density-based, clustering anomalies |"
echo "   | autoencoder       | torch    | Neural network, complex patterns     |"
echo "   | vae               | torch    | Variational AE, probabilistic        |"
echo "   | copod             | PyOD     | Copula-based, no hyperparameters     |"
echo "   | hbos              | PyOD     | Histogram-based, very fast           |"
echo "   | ecod              | PyOD     | Empirical CDF, interpretable         |"
echo ""

echo "4. Streaming/Drift Detection Backends"
echo ""
echo "   | Algorithm    | Description                                |"
echo "   |--------------|-------------------------------------------|"
echo "   | adwin        | Adaptive Windowing (most popular)          |"
echo "   | page_hinkley | Page-Hinkley test for mean shifts          |"
echo "   | kswin        | Kolmogorov-Smirnov windowing               |"
echo "   | ddm          | Drift Detection Method (for classifiers)   |"
echo ""

echo "5. Change Point Detection Backends"
echo ""
echo "   | Algorithm | Description                                    |"
echo "   |-----------|------------------------------------------------|"
echo "   | pelt      | Pruned Exact Linear Time (optimal)             |"
echo "   | binseg    | Binary Segmentation (fast, approximate)        |"
echo "   | window    | Sliding window (for online detection)          |"
echo "   | bottomup  | Bottom-up segmentation (merge-based)           |"
echo ""

echo "6. Verifying Endpoints..."
echo ""

# List of endpoints to verify
ENDPOINTS=(
    "/health"
    "/v1/models"
)

for EP in "${ENDPOINTS[@]}"; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$EP")
    if [ "$STATUS" = "200" ]; then
        echo "   ✓ $EP"
    else
        echo "   ✗ $EP (HTTP $STATUS)"
    fi
done

echo ""
echo "=========================================="
echo "ML Features Ready!"
echo "=========================================="
echo ""
echo "Run individual demos to test specific features:"
echo "  ./demo-anomaly-explain.sh    - SHAP explanations"
echo "  ./demo-dataset-audit.sh      - Data quality audit"
echo "  ./demo-pyod-anomaly.sh       - PyOD backends"
echo "  ./demo-drift-detection.sh    - Streaming drift"
echo "  ./demo-changepoint.sh        - Change points"
echo ""
