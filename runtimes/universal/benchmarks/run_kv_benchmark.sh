#!/bin/bash
# KV Cache Benchmark Runner
# Starts runtime, runs benchmark, stops runtime.
# Usage: ./run_kv_benchmark.sh <model_id> [--n-ctx N]
#
# Examples:
#   ./run_kv_benchmark.sh unsloth/gpt-oss-20b-GGUF:UD-Q6_K_XL
#   ./run_kv_benchmark.sh THUDM/glm-4-9b-GGUF
#   ./run_kv_benchmark.sh Qwen/Qwen3-8B-GGUF

set -uo pipefail
# Note: NOT using set -e — we capture exit codes manually to ensure cleanup runs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$(dirname "$SCRIPT_DIR")"
BASE_URL="http://127.0.0.1:11540"
MODEL="${1:?Usage: $0 <model_id> [extra benchmark args...]}"
shift

cd "$RUNTIME_DIR"

# Kill any existing runtime on 11540
echo "=== Cleaning up ==="
EXISTING=$(lsof -ti :11540 2>/dev/null || true)
if [ -n "$EXISTING" ]; then
    echo "Killing existing process on :11540 (pid $EXISTING)"
    kill $EXISTING 2>/dev/null || true
    sleep 2
fi

# Start runtime in background
echo "=== Starting Universal Runtime ==="
TRANSFORMERS_SKIP_MPS=1 uv run python server.py &
RUNTIME_PID=$!
echo "Runtime PID: $RUNTIME_PID"

# Wait for runtime to be ready
echo -n "Waiting for runtime..."
for i in $(seq 1 30); do
    if curl -sf "$BASE_URL/v1/cache/stats" >/dev/null 2>&1; then
        echo " ready!"
        break
    fi
    if ! kill -0 $RUNTIME_PID 2>/dev/null; then
        echo " FAILED (runtime died)"
        exit 1
    fi
    echo -n "."
    sleep 1
done

if ! curl -sf "$BASE_URL/v1/cache/stats" >/dev/null 2>&1; then
    echo " TIMEOUT"
    kill $RUNTIME_PID 2>/dev/null || true
    exit 1
fi

# Unload any models first
echo "=== Unloading existing models ==="
curl -sf -X POST "$BASE_URL/v1/models/unload" | python3 -m json.tool

# Run benchmark
echo ""
echo "=== Running KV Cache Benchmark: $MODEL ==="
echo ""
uv run python benchmarks/kv_cache_benchmark.py \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    "$@"
BENCH_EXIT=$?

# Cleanup
echo ""
echo "=== Stopping runtime (pid $RUNTIME_PID) ==="
kill $RUNTIME_PID 2>/dev/null || true
wait $RUNTIME_PID 2>/dev/null || true

exit $BENCH_EXIT
