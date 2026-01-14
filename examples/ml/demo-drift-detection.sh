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
RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:${LF_RUNTIME_PORT:-11540}}"

echo "=========================================="
echo "Concept Drift Detection Demo (River)"
echo "=========================================="
echo ""

# Check if server is running
echo "1. Checking server health..."
if ! curl -s "$RUNTIME_URL/health" > /dev/null 2>&1; then
    echo "   ERROR: Server not running at $RUNTIME_URL"
    echo "   Start with: cd runtimes/universal && uv run python server.py"
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

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/detect" \
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
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/detect" \
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

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/detect" \
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

# Stateful streaming
echo "5. Stateful Streaming (Create + Update)"
echo "   Creating a drift detector for streaming updates..."
echo ""

# Create detector
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/create?algorithm=adwin&detector_id=demo-detector")
echo "   Create response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

DETECTOR_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('detector_id', 'demo-detector'))")

# Send some updates
echo "   Sending stable values (mean=10)..."
for i in {1..10}; do
    VALUE=$(python3 -c "import random; print(round(10 + random.uniform(-1, 1), 2))")
    curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/update/$DETECTOR_ID?value=$VALUE" > /dev/null
done
echo "   Sent 10 stable values"

# Get state
RESPONSE=$(curl -s "$RUNTIME_URL/v1/streaming/drift/state/$DETECTOR_ID")
echo ""
echo "   Current state after stable values:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

echo "   Sending drifted values (mean=50)..."
for i in {1..10}; do
    VALUE=$(python3 -c "import random; print(round(50 + random.uniform(-1, 1), 2))")
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/update/$DETECTOR_ID?value=$VALUE")
    DRIFT=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('drift_detected', False))")
    if [ "$DRIFT" = "True" ]; then
        echo "   ** DRIFT DETECTED at value $VALUE **"
    fi
done
echo "   Sent 10 drifted values"

# Final state
RESPONSE=$(curl -s "$RUNTIME_URL/v1/streaming/drift/state/$DETECTOR_ID")
echo ""
echo "   Final state:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Delete detector
echo "6. Cleanup"
curl -s -X DELETE "$RUNTIME_URL/v1/streaming/drift/$DETECTOR_ID" > /dev/null
echo "   Deleted detector: $DETECTOR_ID"
echo ""

# Server metrics simulation
echo "7. Realistic Example: Server Latency Monitoring"
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

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/streaming/drift/detect" \
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
echo "  - Stateful streaming updates"
echo "  - Realistic monitoring scenario"
echo ""
echo "Use cases:"
echo "  - ML model monitoring (detect when to retrain)"
echo "  - Server performance monitoring"
echo "  - Financial time-series analysis"
echo "  - IoT sensor anomaly detection"
