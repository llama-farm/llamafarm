#!/bin/bash
# Demo: Time-Series Forecasting with Chronos-Bolt
#
# This demo shows how to forecast future values using Chronos-Bolt,
# a transformer-based time-series forecasting model.
#
# Phase 11 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-timeseries-forecast.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$SCRIPT_DIR/../../runtimes/universal"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env" 2>/dev/null || true
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Time-Series Forecasting Demo (Chronos-Bolt)${NC}"
echo -e "${BLUE}  Port: ${PORT}${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Health check
echo -e "${YELLOW}Checking Universal Runtime health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: cd runtimes/universal && uv run python server.py"
    exit 1
fi
echo -e "${GREEN}✓ Universal Runtime is healthy${NC}"
echo ""

# Test 1: Linear Trend
echo -e "${BLUE}Test 1: Forecast linear trend${NC}"
echo -e "${YELLOW}Input: 30 days of linear growth (1, 2, 3, ..., 30)${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/timeseries/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "horizon": 7
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'detail' in data:
    print(f\"Error: {data['detail']}\")
    sys.exit(1)
print(f\"Input length: {data['input_length']} days\")
print(f\"Forecast horizon: {data['horizon']} days\")
print(f\"Quantiles used: {data.get('quantiles', [])}\")
print()
print('Forecasts (with confidence intervals):')
for f in data['forecasts']:
    print(f\"  Day {30 + f['step']}: {f['point']:.1f} [{f['lower']:.1f} - {f['upper']:.1f}]\")
"
echo ""

# Test 2: Seasonal Pattern
echo -e "${BLUE}Test 2: Forecast seasonal pattern${NC}"
echo -e "${YELLOW}Input: 21 days with weekly seasonality${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/timeseries/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [10, 12, 14, 15, 13, 11, 10, 11, 13, 15, 16, 14, 12, 11, 12, 14, 16, 17, 15, 13, 12],
    "horizon": 7,
    "quantiles": [0.1, 0.5, 0.9]
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'detail' in data:
    print(f\"Error: {data['detail']}\")
    sys.exit(1)
print(f\"Input length: {data['input_length']} days\")
print()
print('Forecasts for next week:')
for f in data['forecasts']:
    width = f['upper'] - f['lower']
    print(f\"  Day {f['step']}: {f['point']:.1f} (uncertainty: ±{width/2:.1f})\")
"
echo ""

# Test 3: Real-world-like sales data
echo -e "${BLUE}Test 3: Forecast sales-like data${NC}"
echo -e "${YELLOW}Input: 30 days of simulated daily sales${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/timeseries/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [120, 135, 142, 155, 168, 210, 195, 125, 140, 150, 162, 175, 215, 200, 130, 145, 155, 165, 180, 220, 205, 135, 150, 160, 170, 185, 225, 210, 140, 155],
    "horizon": 7,
    "num_samples": 50
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'detail' in data:
    print(f\"Error: {data['detail']}\")
    sys.exit(1)
print('Daily sales forecasts:')
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i, f in enumerate(data['forecasts']):
    day = days[i % 7]
    print(f\"  {day}: \${f['point']:.0f} (range: \${f['lower']:.0f} - \${f['upper']:.0f})\")

# Calculate weekly total
total = sum(f['point'] for f in data['forecasts'])
print(f\"\nProjected weekly total: \${total:.0f}\")
"
echo ""

# Test 4: Custom quantiles
echo -e "${BLUE}Test 4: Forecast with custom confidence levels${NC}"
echo -e "${YELLOW}Using 5th and 95th percentiles for wider intervals${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/timeseries/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [100, 105, 110, 108, 112, 118, 115, 120, 125, 122, 128, 130, 127, 135, 140],
    "horizon": 5,
    "quantiles": [0.05, 0.5, 0.95]
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'detail' in data:
    print(f\"Error: {data['detail']}\")
    sys.exit(1)
print(f\"Using quantiles: {data.get('quantiles', [])}\")
print()
print('Forecasts (90% confidence interval):')
for f in data['forecasts']:
    print(f\"  Step {f['step']}: {f['point']:.1f} [{f['lower']:.1f} - {f['upper']:.1f}]\")
"
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Time-Series Forecasting (Chronos-Bolt) supports:"
echo "  - Probabilistic forecasts with confidence intervals"
echo "  - Configurable forecast horizon"
echo "  - Custom quantile levels for uncertainty"
echo "  - Works with any time-series data"
echo "  - Batch processing for multiple series"
