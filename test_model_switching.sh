#!/bin/bash
# Test model switching in CLI

echo "=== Testing Model Switching ==="
echo ""
echo "This test will:"
echo "1. Send request with 'fast' model (Ollama)"
echo "2. Send request with 'lemon' model (Lemonade port 11534)"
echo "3. Verify requests go to correct endpoints"
echo ""

# Kill existing Lemonade servers
echo "Cleaning up existing Lemonade servers..."
pkill -f "lemonade-server-dev" || true
sleep 2

# Start Lemonade on port 11534
echo "Starting Lemonade on port 11534..."
cd runtimes/lemonade
LEMONADE_MODEL_NAME=lemon bash start.sh > /tmp/lemonade-lemon.log 2>&1 &
LEMON_PID=$!
cd ../..
sleep 5

echo "Lemonade started (PID: $LEMON_PID)"
echo ""

# Test with fast (Ollama) model
echo "=== Test 1: Using 'fast' model (Ollama on :11434) ==="
echo "Expected: Request should go to http://localhost:11434/v1"
echo ""
./lf chat --model fast "Say 'Hello from Ollama'" 2>&1 | head -20
echo ""

# Test with lemon (Lemonade) model
echo "=== Test 2: Using 'lemon' model (Lemonade on :11534) ==="
echo "Expected: Request should go to http://127.0.0.1:11534/api/v1"
echo ""
./lf chat --model lemon "Say 'Hello from Lemonade'" 2>&1 | head -20
echo ""

# Check logs
echo "=== Checking Lemonade logs ==="
echo "Last 10 lines from /tmp/lemonade-lemon.log:"
tail -10 /tmp/lemonade-lemon.log
echo ""

# Cleanup
echo "Cleaning up..."
kill $LEMON_PID 2>/dev/null || true
pkill -f "lemonade-server-dev" || true

echo ""
echo "=== Test Complete ==="
echo "Check server logs for model resolution details"
