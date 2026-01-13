#!/bin/bash
# Demo: Language Detection
#
# This demo shows how to detect the language of text inputs.
# Uses XLM-RoBERTa fine-tuned for language detection.
# Supports 20 languages including English, Spanish, French, German,
# Chinese, Japanese, Arabic, Russian, and more.
#
# Phase 6 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-language-detection.sh

set -e

# Load port from .env if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    source "$SCRIPT_DIR/../../.env" 2>/dev/null || true
fi

PORT=${LF_RUNTIME_PORT:-11545}
BASE_URL="http://localhost:${PORT}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Language Detection Demo${NC}"
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

# Test 1: Detect English
echo -e "${BLUE}Test 1: Detect English text${NC}"
echo -e "${YELLOW}Text: \"Hello, how are you today?\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you today?", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 2: Detect Spanish
echo -e "${BLUE}Test 2: Detect Spanish text${NC}"
echo -e "${YELLOW}Text: \"Hola, ¿cómo estás hoy?\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hola, ¿cómo estás hoy?", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 3: Detect French
echo -e "${BLUE}Test 3: Detect French text${NC}"
echo -e "${YELLOW}Text: \"Bonjour, comment allez-vous?\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour, comment allez-vous?", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 4: Detect German
echo -e "${BLUE}Test 4: Detect German text${NC}"
echo -e "${YELLOW}Text: \"Guten Tag, wie geht es Ihnen?\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "Guten Tag, wie geht es Ihnen?", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 5: Detect Chinese
echo -e "${BLUE}Test 5: Detect Chinese text${NC}"
echo -e "${YELLOW}Text: \"你好，今天天气怎么样？\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，今天天气怎么样？", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 6: Detect Japanese
echo -e "${BLUE}Test 6: Detect Japanese text${NC}"
echo -e "${YELLOW}Text: \"こんにちは、今日はお元気ですか？\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language" \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは、今日はお元気ですか？", "top_k": 3}' | python3 -m json.tool
echo ""

# Test 7: Batch detection
echo -e "${BLUE}Test 7: Batch language detection (multiple texts)${NC}"
echo -e "${YELLOW}Detecting languages for multiple texts at once${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/language/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello world",
      "Bonjour le monde",
      "Hallo Welt",
      "Ciao mondo",
      "Olá mundo"
    ],
    "top_k": 1
  }' | python3 -m json.tool
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Language detection supports 20 languages:"
echo "  Arabic, Bulgarian, German, Greek, English, Spanish, French,"
echo "  Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian,"
echo "  Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese"
