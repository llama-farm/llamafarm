#!/bin/bash
# Demo: Concept Drift Detection (River)
# Phase 14 of Universal Runtime ML Enhancements
#
# Detects when statistical properties of streaming data change.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi

PORT=${1:-8000}
BASE_URL="http://localhost:${PORT}"

echo "=========================================="
echo "Concept Drift Detection Demo (River)"
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

# Generate test data using Python
echo "2. Basic Drift Detection"
echo "   Creating a stream with a sudden mean shift..."
echo ""

# Generate stream with drift: 50 values around mean=10, then 50 values around mean=50
STREAM=$(python3 -c "
import random
random.seed(42)
# First 50 values around mean 10
stream1 = [10 + random.uniform(-1, 1) for _ in range(50)]
# Next 50 values around mean 50 (drift!)
stream2 = [50 + random.uniform(-1, 1) for _ in range(50)]
print(str(stream1 + stream2).replace(' ', ''))
")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/drift" \
    -H "Content-Type: application/json" \
    -d "{
        \"values\": $STREAM,
        \"algorithm\": \"adwin\",
        \"delta\": 0.001
    }")

echo "   Stream: 50 values ~10, then 50 values ~50 (drift at index ~50)"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Comparing algorithms
echo "3. Comparing Drift Detection Algorithms"
echo "   Testing the same stream with different algorithms..."
echo ""

for ALGO in adwin page_hinkley kswin; do
    echo "   Algorithm: $ALGO"
    RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/drift" \
        -H "Content-Type: application/json" \
        -d "{
            \"values\": $STREAM,
            \"algorithm\": \"$ALGO\"
        }")
    DRIFT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_detected', False))")
    POINTS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_points', []))")
    echo "   Drift detected: $DRIFT"
    echo "   Drift points: $POINTS"
    echo ""
done

# Stable stream (no drift)
echo "4. Stable Stream (No Drift Expected)"
echo "   Creating a stream with no drift..."
echo ""

STABLE_STREAM=$(python3 -c "
import random
random.seed(42)
stream = [10 + random.uniform(-0.5, 0.5) for _ in range(100)]
print(str(stream).replace(' ', ''))
")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/drift" \
    -H "Content-Type: application/json" \
    -d "{
        \"values\": $STABLE_STREAM,
        \"algorithm\": \"adwin\"
    }")

echo "   Stream: 100 stable values around mean=10"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Server metrics simulation
echo "5. Realistic Example: Server Latency Monitoring"
echo "   Simulating server latency with sudden increase..."
echo ""

LATENCY_STREAM=$(python3 -c "
import random
random.seed(42)
# Normal latency: 20-30ms
normal = [25 + random.uniform(-5, 5) for _ in range(50)]
# High latency incident: 100-150ms
incident = [125 + random.uniform(-25, 25) for _ in range(30)]
# Back to normal
recovered = [25 + random.uniform(-5, 5) for _ in range(20)]
print(str(normal + incident + recovered).replace(' ', ''))
")

RESPONSE=$(curl -s -X POST "$BASE_URL/v1/ml/analysis/drift" \
    -H "Content-Type: application/json" \
    -d "{
        \"values\": $LATENCY_STREAM,
        \"algorithm\": \"adwin\",
        \"delta\": 0.001
    }")

echo "   Latency pattern:"
echo "   - Normal (index 0-49): ~25ms"
echo "   - Incident (index 50-79): ~125ms"
echo "   - Recovered (index 80-99): ~25ms"
echo ""
echo "   Drift detection result:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key features demonstrated:"
echo "  - Batch drift detection"
echo "  - Multiple algorithms (ADWIN, Page-Hinkley, KSWIN)"
echo "  - Realistic monitoring scenario"
echo ""
echo "Use cases:"
echo "  - ML model monitoring (detect when to retrain)"
echo "  - Server performance monitoring"
echo "  - Financial time-series analysis"
echo "  - IoT sensor anomaly detection"
