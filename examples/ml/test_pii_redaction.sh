#!/bin/bash
# Test PII Redaction (GLiNER) endpoint
#
# This script demonstrates:
# 1. Detecting PII in text
# 2. Redacting sensitive information
# 3. Custom entity types
#
# Usage: ./test_pii_redaction.sh [PORT]
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
echo -e "${BLUE}  PII Redaction (GLiNER) Test${NC}"
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
# Test 1: Detect and Redact PII
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 1: Detect and Redact PII${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

SAMPLE_TEXT="Contact John Smith at john.smith@email.com or call 555-123-4567. His SSN is 123-45-6789 and he lives at 123 Main Street, New York."

echo -e "${YELLOW}Original text:${NC}"
echo "  $SAMPLE_TEXT"
echo ""

echo -e "${YELLOW}Detecting PII...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/redact" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${SAMPLE_TEXT}\"
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'entities' in data:
        entities = data['entities']
        print(f'  Found {len(entities)} PII entity(ies):')
        for e in entities:
            etype = e.get('type', 'UNKNOWN')
            text = e.get('text', '')
            print(f'    - {etype}: \"{text}\"')
    if 'redacted_text' in data:
        print(f'')
        print(f'  Redacted text:')
        print(f'    {data[\"redacted_text\"]}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 2: Custom Entity Types
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 2: Custom Entity Types${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

MEDICAL_TEXT="Patient Jane Doe, DOB 01/15/1985, was diagnosed with Type 2 diabetes. Dr. Robert Johnson prescribed Metformin 500mg."

echo -e "${YELLOW}Medical text:${NC}"
echo "  ${MEDICAL_TEXT:0:80}..."
echo ""

echo -e "${YELLOW}Detecting medical PII...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/redact" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${MEDICAL_TEXT}\",
        \"entity_types\": [\"PERSON\", \"DATE\", \"MEDICAL_CONDITION\", \"MEDICATION\"]
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'entities' in data:
        entities = data['entities']
        print(f'  Found {len(entities)} entity(ies):')
        for e in entities:
            print(f'    - {e.get(\"type\", \"?\")}: \"{e.get(\"text\", \"\")}\"')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
echo ""

# ============================================================================
# Test 3: Financial PII
# ============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Test 3: Financial PII${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

FINANCIAL_TEXT="Transfer $5,000 to account 1234-5678-9012-3456 at Bank of America. Contact support at support@bank.com."

echo -e "${YELLOW}Financial text:${NC}"
echo "  $FINANCIAL_TEXT"
echo ""

echo -e "${YELLOW}Detecting financial PII...${NC}"
RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/nlp/redact" \
    -H "Content-Type: application/json" \
    --max-time 60 \
    -d "{
        \"text\": \"${FINANCIAL_TEXT}\"
    }")

echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'entities' in data:
        entities = data['entities']
        print(f'  Found {len(entities)} entity(ies):')
        for e in entities:
            print(f'    - {e.get(\"type\", \"?\")}: \"{e.get(\"text\", \"\")}\"')
    if 'redacted_text' in data:
        print(f'')
        print(f'  Redacted: {data[\"redacted_text\"]}')
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
echo "API Endpoint: POST /v1/nlp/redact"
echo ""
echo "Request format:"
echo "  {"
echo "    \"text\": \"Text with PII\","
echo "    \"entity_types\": [\"PERSON\", \"EMAIL\", \"PHONE\"]  // Optional"
echo "  }"
echo ""
echo "Response format:"
echo "  {"
echo "    \"entities\": ["
echo "      {\"type\": \"EMAIL\", \"text\": \"john@email.com\", \"start\": 10, \"end\": 25}"
echo "    ],"
echo "    \"redacted_text\": \"Contact [PERSON] at [EMAIL]...\""
echo "  }"
echo ""
echo "Common PII types:"
echo "  PERSON, EMAIL, PHONE, SSN, ADDRESS, DATE, CREDIT_CARD, etc."
echo ""
