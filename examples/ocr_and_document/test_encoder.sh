#!/bin/bash
# Test Encoder models (Embeddings, Classification, Reranking)
#
# This script demonstrates:
# 1. Text embeddings with sentence-transformers
# 2. Text classification (sentiment analysis)
# 3. Document reranking
#
# Usage: ./test_encoder.sh [PORT]
#   PORT defaults to 11540 (Universal Runtime)

set -e

PORT=${1:-11540}
BASE_URL="http://localhost:${PORT}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Universal Runtime Encoder Models Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if server is running
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# ============================================================================
# Test 1: Text Embeddings
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Text Embeddings${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Generating embeddings for sample texts...${NC}"
echo "   Model: sentence-transformers/all-MiniLM-L6-v2"
echo ""

EMBEDDING_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/embeddings" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn canine leaps above a sleepy hound.",
            "Machine learning models are powerful tools."
        ]
    }')

# Check for success
if echo "$EMBEDDING_RESPONSE" | grep -q '"embedding"'; then
    echo -e "${GREEN}✓ Embeddings generated successfully!${NC}"
    echo ""

    # Show embedding stats
    echo "$EMBEDDING_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for item in data.get('data', []):
        emb = item['embedding']
        print(f\"Text {item['index'] + 1}: {len(emb)} dimensions\")
        print(f\"  First 5 values: {emb[:5]}\")

    # Calculate similarity between first two (should be high - similar meaning)
    import math
    e1 = data['data'][0]['embedding']
    e2 = data['data'][1]['embedding']
    e3 = data['data'][2]['embedding']

    def cosine_sim(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(x*x for x in b))
        return dot / (norm_a * norm_b)

    print()
    print(f'Similarity (text 1 vs 2 - similar meaning): {cosine_sim(e1, e2):.4f}')
    print(f'Similarity (text 1 vs 3 - different topic): {cosine_sim(e1, e3):.4f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${RED}Error generating embeddings:${NC}"
    echo "$EMBEDDING_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$EMBEDDING_RESPONSE"
fi
echo ""

# ============================================================================
# Test 2: Text Classification (Sentiment)
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Sentiment Classification${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Classifying sentiment of sample texts...${NC}"
echo "   Model: distilbert-base-uncased-finetuned-sst-2-english"
echo ""

CLASSIFY_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/classify" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "texts": [
            "This product is absolutely amazing! Best purchase ever!",
            "Terrible experience. Would not recommend to anyone.",
            "The weather today is partly cloudy.",
            "I love spending time with my family on weekends."
        ]
    }')

# Check for success
if echo "$CLASSIFY_RESPONSE" | grep -q '"label"'; then
    echo -e "${GREEN}✓ Classification completed successfully!${NC}"
    echo ""

    echo "$CLASSIFY_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    texts = [
        'This product is absolutely amazing! Best purchase ever!',
        'Terrible experience. Would not recommend to anyone.',
        'The weather today is partly cloudy.',
        'I love spending time with my family on weekends.'
    ]
    for item in data.get('data', []):
        idx = item['index']
        print(f'Text: \"{texts[idx][:50]}...\"')
        print(f'  Label: {item[\"label\"]} (confidence: {item[\"score\"]:.4f})')
        print()
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
else
    echo -e "${YELLOW}Classification may have failed or model not available${NC}"
    echo "$CLASSIFY_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$CLASSIFY_RESPONSE"
fi
echo ""

# ============================================================================
# Test 3: Document Reranking
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Document Reranking${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Reranking documents for a query...${NC}"
echo "   Model: cross-encoder/ms-marco-MiniLM-L-6-v2"
echo "   Query: What are the benefits of exercise?"
echo ""

RERANK_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/rerank" \
    -H "Content-Type: application/json" \
    --max-time 120 \
    -d '{
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "query": "What are the benefits of exercise?",
        "documents": [
            "The stock market showed mixed results today.",
            "Regular physical activity improves cardiovascular health and reduces stress.",
            "Python is a popular programming language.",
            "Exercise helps maintain healthy weight and boosts mental well-being.",
            "The recipe calls for two cups of flour."
        ],
        "top_k": 3,
        "return_documents": true
    }')

# Check for success
if echo "$RERANK_RESPONSE" | grep -q '"relevance_score"'; then
    echo -e "${GREEN}✓ Reranking completed successfully!${NC}"
    echo ""

    echo "Top 3 most relevant documents:"
    echo "---"
    echo "$RERANK_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for rank, item in enumerate(data.get('data', []), 1):
        print(f'{rank}. Score: {item[\"relevance_score\"]:.4f}')
        print(f'   \"{item.get(\"document\", \"N/A\")[:70]}...\"')
        print()
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    echo "---"
else
    echo -e "${YELLOW}Reranking may have failed or model not available${NC}"
    echo "$RERANK_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RERANK_RESPONSE"
fi
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
