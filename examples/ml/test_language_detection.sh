#!/bin/bash
# Test Language Detection endpoint
#
# This script demonstrates:
# 1. Detecting language of text in multiple languages
# 2. Showing confidence scores
#
# Usage: ./test_language_detection.sh [PORT]
#   PORT defaults to LF_RUNTIME_PORT from .env (fallback: 11540)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Load port from .env file
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env"
fi


PORT=${1:-${LF_RUNTIME_PORT:-11540}}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Language Detection Test${NC}"
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

# ============================================================================
# Test 1: English Detection
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Language Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Test samples in different languages
declare -a SAMPLES=(
    "English|Hello, this is a test sentence in English. The weather is nice today."
    "Spanish|Hola, esta es una oración de prueba en español. El clima está agradable hoy."
    "French|Bonjour, ceci est une phrase de test en français. Le temps est agréable aujourd'hui."
    "German|Hallo, dies ist ein Testsatz auf Deutsch. Das Wetter ist heute schön."
    "Chinese|你好，这是一个中文测试句子。今天天气很好。"
    "Japanese|こんにちは、これは日本語のテスト文です。今日は天気がいいです。"
)

for SAMPLE in "${SAMPLES[@]}"; do
    IFS='|' read -r EXPECTED TEXT <<< "$SAMPLE"
    echo -e "${YELLOW}Testing ${EXPECTED} text...${NC}"

    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/identify-language" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"text\": \"${TEXT}\"
        }")

    echo "$RESPONSE" | python3 -c "
import sys, json
expected = '${EXPECTED}'
try:
    data = json.load(sys.stdin)
    if 'language' in data:
        lang = data['language']
        score = data.get('score', 0)
        match = '✓' if expected.lower() in lang.lower() else '?'
        print(f'  {match} Detected: {lang} ({score:.2%} confidence)')
    else:
        print(f'  Response: {data}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    echo ""
done

# ============================================================================
# Test 2: Short Text Detection
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Short Text Detection${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

echo -e "${YELLOW}Testing short texts...${NC}"

declare -a SHORT_SAMPLES=(
    "Hello world"
    "Bonjour"
    "Guten Tag"
    "Hola"
)

for TEXT in "${SHORT_SAMPLES[@]}"; do
    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/identify-language" \
        -H "Content-Type: application/json" \
        --max-time 60 \
        -d "{
            \"text\": \"${TEXT}\"
        }")

    echo "$RESPONSE" | python3 -c "
import sys, json
text = '${TEXT}'
try:
    data = json.load(sys.stdin)
    if 'language' in data:
        lang = data['language']
        score = data.get('score', 0)
        print(f'  \"{text}\" → {lang} ({score:.0%})')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
done
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Test Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "API Endpoint: POST /v1/nlp/identify-language"
echo ""
echo "Request format:"
echo "  {"
echo "    \"text\": \"Text to identify\""
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"language\": \"English\","
echo "    \"score\": 0.98"
echo "  }"
echo ""
