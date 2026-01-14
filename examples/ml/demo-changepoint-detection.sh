#!/bin/bash
# Demo: Change Point Detection (Ruptures)
# Phase 12 of Universal Runtime ML Enhancements
#
# Detects structural changes in time-series data using various algorithms.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi
RUNTIME_URL="${RUNTIME_URL:-http://127.0.0.1:${LF_RUNTIME_PORT:-11540}}"

echo "=========================================="
echo "Change Point Detection Demo (Ruptures)"
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

# Basic change point detection
echo "2. Basic Change Point Detection"
echo "   Detecting changes in a step signal..."
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints" \
    -H "Content-Type: application/json" \
    -d '{
        "values": [1,1,1,1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5,5,5, 2,2,2,2,2,2,2,2,2,2],
        "algorithm": "pelt",
        "model": "rbf"
    }')

echo "   Signal: [1,1,1,1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5,5,5, 2,2,2,2,2,2,2,2,2,2]"
echo "   (Three segments: values at 1, 5, and 2)"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Different algorithms
echo "3. Comparing Algorithms"
echo "   Using different algorithms on the same signal..."
echo ""

for ALGO in pelt binseg window bottomup; do
    echo "   Algorithm: $ALGO"
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints" \
        -H "Content-Type: application/json" \
        -d "{
            \"values\": [1,1,1,1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5,5,5, 2,2,2,2,2,2,2,2,2,2],
            \"algorithm\": \"$ALGO\",
            \"model\": \"rbf\"
        }")
    CHANGE_POINTS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('change_points', []))")
    N_SEGMENTS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('n_segments', 0))")
    echo "   Change points: $CHANGE_POINTS"
    echo "   Segments: $N_SEGMENTS"
    echo ""
done

# Specifying exact number of change points
echo "4. Exact Number of Change Points"
echo "   Requesting exactly 2 change points..."
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints" \
    -H "Content-Type: application/json" \
    -d '{
        "values": [1,1,1,1,1, 3,3,3,3,3, 5,5,5,5,5, 2,2,2,2,2],
        "n_changepoints": 2,
        "algorithm": "binseg",
        "model": "l2"
    }')

echo "   Signal: [1,1,1,1,1, 3,3,3,3,3, 5,5,5,5,5, 2,2,2,2,2]"
echo "   (Requested: 2 change points)"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Penalty parameter
echo "5. Penalty Parameter Effect"
echo "   Testing different penalty values..."
echo ""

for PENALTY in 1 10 100; do
    echo "   Penalty: $PENALTY"
    RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints" \
        -H "Content-Type: application/json" \
        -d "{
            \"values\": [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4],
            \"penalty\": $PENALTY,
            \"algorithm\": \"pelt\",
            \"model\": \"l2\"
        }")
    CHANGE_POINTS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('change_points', []))")
    echo "   Change points: $CHANGE_POINTS"
done
echo ""

# Batch processing
echo "6. Batch Change Point Detection"
echo "   Processing multiple time-series at once..."
echo ""
RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "series": [
            [1,1,1,1,1, 5,5,5,5,5],
            [2,2,2,2,2, 8,8,8,8,8, 3,3,3,3,3],
            [1,2,3,4,5,6,7,8,9,10]
        ],
        "algorithm": "pelt",
        "model": "rbf"
    }')

echo "   Series 1: [1,1,1,1,1, 5,5,5,5,5] (one step change)"
echo "   Series 2: [2,2,2,2,2, 8,8,8,8,8, 3,3,3,3,3] (two step changes)"
echo "   Series 3: [1,2,3,4,5,6,7,8,9,10] (linear trend)"
echo ""
echo "   Response:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

# Realistic example: Server metrics
echo "7. Realistic Example: Server Metrics Anomaly"
echo "   Detecting when server behavior changed..."
echo ""

# Simulated CPU usage: normal -> high load -> normal
CPU_DATA='[20,22,19,21,20,23,18,21,20,22,85,88,92,87,90,89,91,88,25,23,21,24,22,20,21,23]'

RESPONSE=$(curl -s -X POST "$RUNTIME_URL/v1/timeseries/changepoints" \
    -H "Content-Type: application/json" \
    -d "{
        \"values\": $CPU_DATA,
        \"algorithm\": \"pelt\",
        \"model\": \"rbf\"
    }")

echo "   Simulated CPU usage over time:"
echo "   $CPU_DATA"
echo "   (Normal ~20%, spike to ~90%, return to normal)"
echo ""
echo "   Detected change points:"
echo "$RESPONSE" | python3 -m json.tool
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Key features demonstrated:"
echo "  - Basic change point detection"
echo "  - Multiple algorithms (pelt, binseg, window, bottomup)"
echo "  - Multiple cost models (rbf, l1, l2, normal, ar)"
echo "  - Exact number of change points"
echo "  - Penalty parameter for sensitivity control"
echo "  - Batch processing"
echo "  - Realistic server metrics example"
