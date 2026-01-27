#!/bin/bash
# Universal RAG API Demo
#
# This script demonstrates the universal_rag strategy using raw API calls.
# It shows how to ingest and query documents without any parser configuration.
#
# Prerequisites:
#   - LlamaFarm server running on port 8000
#   - Universal runtime running
#
# Usage:
#   bash examples/universal_rag/demo_api.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load port from .env if available
ENV_FILE="$SCRIPT_DIR/../../.env"
if [ -f "$ENV_FILE" ]; then
  PORT=$(grep -E "^PORT=" "$ENV_FILE" | cut -d= -f2)
fi
PORT="${PORT:-8000}"
API_BASE="${API_BASE:-http://localhost:$PORT}"

NAMESPACE="${NAMESPACE:-examples}"
PROJECT="${PROJECT:-universal_rag_demo}"

echo "=== Universal RAG API Demo ==="
echo "API: $API_BASE"
echo "Project: $NAMESPACE/$PROJECT"
echo ""

# Test 1: Check server health
echo "Test 1: Check server health"
echo "----------------------------"
response=$(curl -s "$API_BASE/health" 2>&1)
echo "Response: $response"
echo ""

# Test 2: List available data processing strategies
echo "Test 2: List data processing strategies"
echo "----------------------------------------"
echo "Request: GET $API_BASE/v1/projects/$NAMESPACE/$PROJECT/rag/strategies"
response=$(curl -s "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/rag/strategies" 2>&1)
if echo "$response" | python3 -c "import sys, json; d=json.load(sys.stdin); print('Available:', [s['name'] for s in d.get('strategies', [])])" 2>/dev/null; then
  :
else
  echo "Response: $response"
fi
echo ""

# Test 3: Create a dataset (uses universal_rag by default when no strategy specified)
echo "Test 3: Create dataset with universal_rag"
echo "------------------------------------------"
DATASET_NAME="api_demo_$(date +%s)"
echo "Request: POST $API_BASE/v1/datasets"
response=$(curl -s -X POST "$API_BASE/v1/datasets" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"$DATASET_NAME\",
    \"namespace\": \"$NAMESPACE\",
    \"project\": \"$PROJECT\",
    \"database\": \"universal_db\"
  }" 2>&1)
echo "Response: $response"
echo ""

# Test 4: Upload a document
echo "Test 4: Upload document"
echo "-----------------------"
# Create a temporary test document
TMP_DOC=$(mktemp).txt
cat > "$TMP_DOC" << 'EOF'
Universal RAG Test Document

This is a test document for the Universal RAG demo. It contains information
about the Universal RAG system:

1. UniversalParser: Handles 90%+ of document formats including PDF, DOCX, MD, TXT
2. UniversalExtractor: Extracts keywords, entities, and metadata automatically
3. Semantic Chunking: Intelligent text splitting that respects content boundaries

The universal_rag strategy is used automatically when no data_processing_strategy
is specified in the dataset configuration.
EOF

echo "Request: POST $API_BASE/v1/datasets/$DATASET_NAME/upload"
response=$(curl -s -X POST "$API_BASE/v1/datasets/$DATASET_NAME/upload" \
  -F "file=@$TMP_DOC;filename=test_doc.txt" 2>&1)
echo "Response: $response"
rm -f "$TMP_DOC"
echo ""

# Test 5: Process the dataset
echo "Test 5: Process dataset (universal_rag)"
echo "----------------------------------------"
echo "Request: POST $API_BASE/v1/datasets/$DATASET_NAME/process"
response=$(curl -s -X POST "$API_BASE/v1/datasets/$DATASET_NAME/process" \
  -H "Content-Type: application/json" 2>&1)
echo "Response: $response"
echo ""

# Wait for processing
echo "Waiting for processing to complete..."
sleep 3
echo ""

# Test 6: Query the RAG database
echo "Test 6: RAG Query"
echo "-----------------"
echo "Request: POST $API_BASE/v1/rag/query"
response=$(curl -s -X POST "$API_BASE/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does UniversalParser handle?",
    "database": "universal_db",
    "top_k": 3,
    "include_metadata": true
  }' 2>&1)

if echo "$response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'results' in d:
    for i, r in enumerate(d['results'], 1):
        print(f'Result {i}:')
        print(f'  Content: {r.get(\"content\", \"N/A\")[:100]}...')
        meta = r.get('metadata', {})
        print(f'  Metadata: chunk_index={meta.get(\"chunk_index\")}, document_name={meta.get(\"document_name\")}')
        print()
else:
    print(d)
" 2>/dev/null; then
  :
else
  echo "Response: $response"
fi
echo ""

# Test 7: Chat with RAG context
echo "Test 7: Chat with RAG context"
echo "-----------------------------"
echo "Request: POST $API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions"
response=$(curl -s -X POST "$API_BASE/v1/projects/$NAMESPACE/$PROJECT/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-No-Session: true" \
  -d "{
    \"messages\": [{\"role\": \"user\", \"content\": \"What is semantic chunking?\"}],
    \"rag\": {
      \"enabled\": true,
      \"database\": \"universal_db\",
      \"top_k\": 3
    },
    \"max_tokens\": 150
  }" 2>&1)

if echo "$response" | grep -q "choices"; then
  content=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "$response")
  echo "Response: $content"
else
  echo "Response: $response"
fi
echo ""

echo "=== Demo Complete ==="
echo ""
echo "Key takeaways:"
echo "1. No data_processing_strategy specified = universal_rag is used automatically"
echo "2. UniversalParser handles multiple document formats"
echo "3. UniversalExtractor adds rich metadata to chunks"
echo "4. RAG queries return documents with metadata"
echo ""
echo "Dataset '$DATASET_NAME' created. Clean up when finished."
