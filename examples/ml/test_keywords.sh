#!/bin/bash
# Test Keyword/Keyphrase Extraction endpoint
#
# This script demonstrates:
# 1. Extracting keywords from text
# 2. Extracting keyphrases
# 3. Configuring number of results
#
# Usage: ./test_keywords.sh [PORT]
#   PORT defaults to 11540 (Universal Runtime)

set -e

PORT=${1:-11540}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Keyword/Keyphrase Extraction Test${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check health
echo -e "${YELLOW}Checking server health...${NC}"
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Universal Runtime not running on port ${PORT}${NC}"
    echo "Start it with: nx start universal-runtime"
    exit 1
fi
echo -e "${GREEN}✓ Server is healthy${NC}"
echo ""

# Sample text for testing
SAMPLE_TEXT="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. Deep learning, a specialized form of machine learning, uses neural networks with multiple layers to process complex data patterns. Applications include natural language processing, computer vision, and autonomous vehicles."

# ============================================================================
# Test 1: Extract Keywords
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Extract Keywords${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Sample text:${NC}"
echo "  ${SAMPLE_TEXT:0:100}..."
echo ""

echo -e "${YELLOW}Extracting top 10 keywords...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/keywords" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${SAMPLE_TEXT}\",
        \"top_n\": 10
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'keywords' in data:
        keywords = data['keywords']
        print('  Keywords:')
        for kw in keywords:
            if isinstance(kw, dict):
                print(f'    - {kw.get(\"keyword\", kw)}: {kw.get(\"score\", 0):.4f}')
            else:
                print(f'    - {kw}')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Extract Keyphrases
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Extract Keyphrases${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Extracting top 5 keyphrases...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/keywords" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${SAMPLE_TEXT}\",
        \"top_n\": 5,
        \"keyphrase_ngram_range\": [1, 3]
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'keywords' in data:
        keywords = data['keywords']
        print('  Keyphrases:')
        for kw in keywords:
            if isinstance(kw, dict):
                print(f'    - \"{kw.get(\"keyword\", kw)}\" ({kw.get(\"score\", 0):.4f})')
            else:
                print(f'    - \"{kw}\"')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 3: Different Text Domains
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Different Text Domains${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Medical text
MEDICAL_TEXT="The patient presented with acute myocardial infarction and was administered aspirin and heparin. Echocardiography revealed reduced ejection fraction. Coronary angiography showed significant stenosis in the left anterior descending artery."

echo -e "${YELLOW}Medical text keywords...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/keywords" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${MEDICAL_TEXT}\",
        \"top_n\": 5
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'keywords' in data:
        kws = [kw.get('keyword', kw) if isinstance(kw, dict) else kw for kw in data['keywords'][:5]]
        print(f'  Top keywords: {', '.join(kws)}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# Legal text
LEGAL_TEXT="The plaintiff alleges breach of contract and seeks compensatory damages. The defendant filed a motion to dismiss citing lack of jurisdiction. The court will hear oral arguments on the preliminary injunction."

echo -e "${YELLOW}Legal text keywords...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/keywords" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${LEGAL_TEXT}\",
        \"top_n\": 5
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'keywords' in data:
        kws = [kw.get('keyword', kw) if isinstance(kw, dict) else kw for kw in data['keywords'][:5]]
        print(f'  Top keywords: {', '.join(kws)}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoint: POST /v1/nlp/keywords"
echo ""
echo "Request format:"
echo "  {"
echo "    \"text\": \"Your text here\","
echo "    \"top_n\": 10,  // Optional, default 10"
echo "    \"keyphrase_ngram_range\": [1, 2]  // Optional"
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"keywords\": ["
echo "      {\"keyword\": \"machine learning\", \"score\": 0.85},"
echo "      ..."
echo "    ]"
echo "  }"
echo ""
