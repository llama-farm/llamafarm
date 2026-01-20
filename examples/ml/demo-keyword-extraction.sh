#!/bin/bash
# Demo: Keyword/Keyphrase Extraction
#
# This demo shows how to extract keywords and keyphrases from text.
# Uses sentence embeddings to find the most relevant n-grams.
#
# Phase 7 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-keyword-extraction.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
echo -e "${BLUE}  Keyword/Keyphrase Extraction Demo${NC}"
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

# Test 1: Technical document
echo -e "${BLUE}Test 1: Extract keywords from technical document${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. Deep learning, a further subset, uses neural networks with multiple layers to model complex patterns in data. Common applications include natural language processing, computer vision, and recommendation systems.",
    "top_k": 10,
    "diversity": 0.5
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Keywords found:', data['count'])
print()
for kw in data['keywords']:
    bar = '█' * int(kw['score'] * 20)
    print(f\"  {kw['keyword']:30s} {kw['score']:.3f} {bar}\")
"
echo ""

# Test 2: News article
echo -e "${BLUE}Test 2: Extract keywords from news article${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve announced a 25 basis point interest rate increase today, citing persistent inflation concerns. Stock markets reacted negatively, with the S&P 500 dropping 2% and technology stocks leading the decline. Economists predict continued monetary tightening through the end of the fiscal year.",
    "top_k": 8,
    "diversity": 0.3
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Keywords found:', data['count'])
print()
for kw in data['keywords']:
    bar = '█' * int(kw['score'] * 20)
    print(f\"  {kw['keyword']:30s} {kw['score']:.3f} {bar}\")
"
echo ""

# Test 3: Different n-gram ranges
echo -e "${BLUE}Test 3: Extract only bigrams and trigrams${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Natural language processing enables computers to understand human language. Text classification and sentiment analysis are common NLP tasks. Named entity recognition identifies people, places, and organizations in text.",
    "top_k": 8,
    "ngram_range": [2, 3],
    "diversity": 0.5
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Keyphrases found (2-3 words):', data['count'])
print()
for kw in data['keywords']:
    bar = '█' * int(kw['score'] * 20)
    print(f\"  {kw['keyword']:30s} {kw['score']:.3f} {bar}\")
"
echo ""

# Test 4: High diversity
echo -e "${BLUE}Test 4: Extract diverse keywords (diversity=0.9)${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Python programming language is popular for data science. Python libraries like NumPy and Pandas help with data analysis. Python is also used for web development with frameworks like Django and Flask. Many Python developers work on machine learning projects.",
    "top_k": 8,
    "diversity": 0.9
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print('Diverse keywords:', data['count'])
print()
for kw in data['keywords']:
    bar = '█' * int(kw['score'] * 20)
    print(f\"  {kw['keyword']:30s} {kw['score']:.3f} {bar}\")
"
echo ""

# Test 5: Batch extraction
echo -e "${BLUE}Test 5: Batch keyword extraction (multiple documents)${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/ml/nlp/keywords/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Cloud computing enables on-demand access to computing resources over the internet.",
      "Blockchain technology provides decentralized and secure transaction recording.",
      "Quantum computing uses quantum mechanics to perform complex calculations."
    ],
    "top_k": 5
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Processed {data['total_count']} documents\")
print()
for i, doc in enumerate(data['data'], 1):
    print(f'Document {i}:')
    for kw in doc['keywords'][:3]:
        print(f\"  - {kw['keyword']} ({kw['score']:.2f})\")
    print()
"

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Keyword extraction finds the most relevant terms in your documents."
echo "Use the diversity parameter to control redundancy in results."
