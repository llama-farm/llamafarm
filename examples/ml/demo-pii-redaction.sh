#!/bin/bash
# Demo: PII Detection and Redaction
#
# This demo shows how to detect and redact Personally Identifiable Information (PII).
# Uses GLiNER for zero-shot entity detection plus regex patterns.
#
# Phase 8 of Universal Runtime ML Enhancements.
#
# Usage: ./demo-pii-redaction.sh

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
echo -e "${BLUE}  PII Detection and Redaction Demo${NC}"
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

# Test 1: Basic PII Detection
echo -e "${BLUE}Test 1: Detect PII in text${NC}"
echo -e "${YELLOW}Input: \"Contact John Smith at john.smith@acme.com or call 555-123-4567\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/pii-detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact John Smith at john.smith@acme.com or call 555-123-4567",
    "threshold": 0.5
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Found {data['entity_count']} PII entities:\")
for e in data['entities']:
    print(f\"  - '{e['text']}' ({e['label']}) score={e['score']:.2f}\")
"
echo ""

# Test 2: Basic Redaction
echo -e "${BLUE}Test 2: Redact PII from text${NC}"
echo -e "${YELLOW}Input: \"My SSN is 123-45-6789 and email is test@example.com\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/pii-redact" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My SSN is 123-45-6789 and email is test@example.com",
    "replacement": "[REDACTED]"
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Original had {data['entity_count']} PII entities\")
print(f\"Redacted: {data['redacted_text']}\")
"
echo ""

# Test 3: Custom Replacement Patterns
echo -e "${BLUE}Test 3: Custom per-type replacement patterns${NC}"
echo -e "${YELLOW}Using different replacements for different PII types${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/pii-redact" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Call Jane Doe at 555-987-6543 or email jane@company.org. Her IP is 192.168.1.100",
    "replacement_map": {
      "person": "[NAME]",
      "email": "[EMAIL]",
      "phone": "[PHONE]",
      "ip_address": "[IP]"
    }
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Redacted ({data['entity_count']} entities):\")
print(f\"  {data['redacted_text']}\")
print()
print('Entity details:')
for e in data['entities']:
    print(f\"  - '{e['text']}' -> [{e['label'].upper()}]\")
"
echo ""

# Test 4: Credit Card Detection
echo -e "${BLUE}Test 4: Detect credit card numbers${NC}"
echo -e "${YELLOW}Input: \"Payment card: 4111-1111-1111-1111, expiry 12/25\"${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/pii-detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Payment card: 4111-1111-1111-1111, expiry 12/25",
    "threshold": 0.3
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Detected {data['entity_count']} entities:\")
for e in data['entities']:
    print(f\"  - '{e['text']}' ({e['label']})\")
"
echo ""

# Test 5: Medical/Healthcare PII
echo -e "${BLUE}Test 5: Detect healthcare-related PII${NC}"
echo -e "${YELLOW}Using custom entity types for healthcare data${NC}"
echo ""

curl -s -X POST "${BASE_URL}/v1/text/pii-detect" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Robert Johnson, DOB 1985-03-15, MRN 12345678, diagnosed with hypertension",
    "entity_types": ["person", "date of birth", "medical record number", "medical condition"],
    "threshold": 0.4
  }' | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"Found {data['entity_count']} healthcare PII entities:\")
for e in data['entities']:
    print(f\"  - '{e['text']}' ({e['label']}) score={e['score']:.2f}\")
"
echo ""

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Demo Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "PII Detection supports:"
echo "  - Standard PII: names, emails, phones, SSN, credit cards, addresses"
echo "  - Custom entity types for domain-specific PII"
echo "  - Configurable replacement patterns per entity type"
echo "  - Regex patterns for high-precision detection of common formats"
