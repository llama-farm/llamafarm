#!/usr/bin/env bash
# Demo: Time-Series Forecasting with LlamaFarm
#
# This demo showcases:
# 1. Classical forecasting with ARIMA (Darts)
# 2. Zero-shot forecasting with Chronos (Amazon foundation model)
# 3. Model persistence and versioning
#
# Prerequisites:
# - Universal Runtime running on port 11540

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           LlamaFarm Time-Series Forecasting Demo             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration - Use LlamaFarm API (port 8005) for production usage
# Falls back to Universal Runtime (port 11545) for local testing
RUNTIME_URL="${LLAMAFARM_URL:-${RUNTIME_URL:-http://127.0.0.1:8005}}"

# Helper function
check_runtime() {
    echo -e "${YELLOW}Checking Universal Runtime availability...${NC}"
    if curl -s "${RUNTIME_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Universal Runtime is running at ${RUNTIME_URL}${NC}"
        return 0
    else
        echo -e "${RED}✗ Universal Runtime not available at ${RUNTIME_URL}${NC}"
        echo -e "${YELLOW}Start it with: nx start universal-runtime${NC}"
        return 1
    fi
}

# Check runtime
if ! check_runtime; then
    exit 1
fi

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 1: List available backends${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "GET ${RUNTIME_URL}/v1/timeseries/backends"
curl -s "${RUNTIME_URL}/v1/timeseries/backends" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 2: Train ARIMA model on sample data${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

# Generate sample training data (30 days of daily data)
TRAIN_DATA='[
  {"timestamp": "2024-01-01", "value": 100},
  {"timestamp": "2024-01-02", "value": 102},
  {"timestamp": "2024-01-03", "value": 105},
  {"timestamp": "2024-01-04", "value": 103},
  {"timestamp": "2024-01-05", "value": 108},
  {"timestamp": "2024-01-06", "value": 110},
  {"timestamp": "2024-01-07", "value": 112},
  {"timestamp": "2024-01-08", "value": 115},
  {"timestamp": "2024-01-09", "value": 118},
  {"timestamp": "2024-01-10", "value": 120},
  {"timestamp": "2024-01-11", "value": 122},
  {"timestamp": "2024-01-12", "value": 125},
  {"timestamp": "2024-01-13", "value": 128},
  {"timestamp": "2024-01-14", "value": 130},
  {"timestamp": "2024-01-15", "value": 132},
  {"timestamp": "2024-01-16", "value": 135},
  {"timestamp": "2024-01-17", "value": 138},
  {"timestamp": "2024-01-18", "value": 140},
  {"timestamp": "2024-01-19", "value": 142},
  {"timestamp": "2024-01-20", "value": 145},
  {"timestamp": "2024-01-21", "value": 148},
  {"timestamp": "2024-01-22", "value": 150},
  {"timestamp": "2024-01-23", "value": 152},
  {"timestamp": "2024-01-24", "value": 155},
  {"timestamp": "2024-01-25", "value": 158},
  {"timestamp": "2024-01-26", "value": 160},
  {"timestamp": "2024-01-27", "value": 162},
  {"timestamp": "2024-01-28", "value": 165},
  {"timestamp": "2024-01-29", "value": 168},
  {"timestamp": "2024-01-30", "value": 170}
]'

echo "POST ${RUNTIME_URL}/v1/timeseries/fit"
echo "Training ARIMA model with 30 data points..."
curl -s -X POST "${RUNTIME_URL}/v1/timeseries/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-arima\",
        \"backend\": \"arima\",
        \"data\": ${TRAIN_DATA},
        \"frequency\": \"D\",
        \"description\": \"Demo ARIMA model for daily forecasting\"
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3: Generate predictions (7-day forecast)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "POST ${RUNTIME_URL}/v1/timeseries/predict"
curl -s -X POST "${RUNTIME_URL}/v1/timeseries/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "demo-arima",
        "horizon": 7,
        "confidence_level": 0.95
    }' | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4: List saved models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "GET ${RUNTIME_URL}/v1/timeseries/models"
curl -s "${RUNTIME_URL}/v1/timeseries/models" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 5: Train Exponential Smoothing model (versioned)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "POST ${RUNTIME_URL}/v1/timeseries/fit"
echo "Training Exponential Smoothing model (overwrite=false for versioning)..."
curl -s -X POST "${RUNTIME_URL}/v1/timeseries/fit" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"demo-exp\",
        \"backend\": \"exponential_smoothing\",
        \"data\": ${TRAIN_DATA},
        \"frequency\": \"D\",
        \"overwrite\": false,
        \"description\": \"Demo ExponentialSmoothing model v1\"
    }" | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 6: Load model using -latest suffix${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "POST ${RUNTIME_URL}/v1/timeseries/load"
curl -s -X POST "${RUNTIME_URL}/v1/timeseries/load" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "demo-exp-latest"
    }' | python3 -m json.tool
echo

echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 7: Cleanup - Delete demo models${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo

echo "DELETE ${RUNTIME_URL}/v1/timeseries/models/demo-arima"
curl -s -X DELETE "${RUNTIME_URL}/v1/timeseries/models/demo-arima" | python3 -m json.tool
echo

echo "DELETE ${RUNTIME_URL}/v1/timeseries/models/demo-exp"
curl -s -X DELETE "${RUNTIME_URL}/v1/timeseries/models/demo-exp" | python3 -m json.tool
echo

echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Demo Complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${YELLOW}Summary:${NC}"
echo "• Listed available backends (arima, exponential_smoothing, theta, chronos, chronos-bolt)"
echo "• Trained ARIMA model on 30 days of daily data"
echo "• Generated 7-day forecast with 95% confidence intervals"
echo "• Demonstrated versioned model creation (overwrite=false)"
echo "• Loaded model using -latest suffix"
echo "• Cleaned up demo models"
echo
echo -e "${BLUE}Note: Chronos zero-shot forecasting requires running the runtime with GPU${NC}"
echo -e "${BLUE}      or sufficient CPU memory for the foundation model.${NC}"
