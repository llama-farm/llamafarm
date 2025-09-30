#!/bin/bash
# Script to check RAG container logs for task registration information

echo "=== RAG Container Task Registration Check ==="
echo

# Check if RAG container is running
if ! docker ps --format "table {{.Names}}" | grep -q "llamafarm-rag"; then
    echo "❌ RAG container (llamafarm-rag) is not running"
    echo "Start it with: lf start"
    exit 1
fi

echo "✓ RAG container is running"
echo

# Show recent logs
echo "Recent RAG container logs:"
echo "=========================="
docker logs --tail 50 llamafarm-rag

echo
echo "=== Filtering for task-related messages ==="
docker logs llamafarm-rag 2>&1 | grep -i -E "(task|register|import|celery)" | tail -20

echo
echo "=== Checking for specific RAG task mentions ==="
docker logs llamafarm-rag 2>&1 | grep -E "rag\.(search|ingest|query|handle|batch)" | tail -10

echo
echo "=== Live log monitoring (Ctrl+C to stop) ==="
echo "Watching for new task registration messages..."
docker logs -f llamafarm-rag 2>&1 | grep --line-buffered -i -E "(task|register|import|rag\.)"
