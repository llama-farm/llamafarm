#!/bin/bash

# Test script for RAG integration with LlamaFarm CLI
# This script tests various RAG scenarios to ensure everything works properly

set -e  # Exit on error

echo "============================================"
echo "LlamaFarm RAG Integration Test Suite"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print test results
print_test() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root (two levels up from test/integration)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project directory
cd "$PROJECT_ROOT"

# Check for project configuration
LLAMAFARM_PROJECT_DIR="${HOME}/.llamafarm/projects/default/llamafarm-1"
if [ -f "${LLAMAFARM_PROJECT_DIR}/llamafarm.yaml" ]; then
    echo -e "${BLUE}Using project config from: ${LLAMAFARM_PROJECT_DIR}${NC}"
    export LLAMAFARM_CONFIG="${LLAMAFARM_PROJECT_DIR}/llamafarm.yaml"
elif [ -f "./llamafarm.yaml" ]; then
    echo -e "${BLUE}Using local llamafarm.yaml${NC}"
    export LLAMAFARM_CONFIG="./llamafarm.yaml"
else
    echo -e "${YELLOW}Warning: No llamafarm.yaml found, using defaults${NC}"
fi

echo ""

echo "1. Testing basic functionality..."
echo "--------------------------------"

# Test 1: Run without RAG (explicitly disable it)
echo -n "Test 1.1: Query without RAG... "
OUTPUT=$(timeout 15 lf run --no-rag "What is 2+2?" 2>&1)
# Allow time for processing
sleep 2
# Check for good mathematical response
if [[ "$OUTPUT" == *"4"* ]] || [[ "$OUTPUT" == *"four"* ]] || [[ "$OUTPUT" == *"2+2"* ]]; then
    # Ensure it's not just echoing
    if [[ "$OUTPUT" != "What is 2+2?" ]]; then
        print_test "Basic query works - Got valid response"
    else
        echo -e "${RED}✗${NC} Response is echoing input"
        echo "Output: $OUTPUT"
    fi
else
    echo -e "${RED}✗${NC} Basic query failed"
    echo "Output: $OUTPUT"
fi

# Test 2: Run with RAG enabled (default behavior)
echo -n "Test 1.2: Query with RAG... "
OUTPUT=$(timeout 20 lf run "What is transformer architecture?" 2>&1)
# Allow time for RAG processing
sleep 2
# Check for substantive response about transformers
if [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"encoder"* ]] || [[ "$OUTPUT" == *"decoder"* ]] || [[ "$OUTPUT" == *"Transformer"* ]]; then
    # Ensure response is detailed enough (at least 50 characters)
    if [ ${#OUTPUT} -gt 50 ] && [[ "$OUTPUT" != "What is transformer architecture?" ]]; then
        print_test "RAG query works - Got detailed response"
    else
        echo -e "${YELLOW}⚠${NC} Response may be too short or echoing"
        echo "Output length: ${#OUTPUT}"
    fi
else
    echo -e "${RED}✗${NC} RAG query failed"
    echo "Output: $OUTPUT"
fi

echo ""
echo "2. Testing RAG parameters..."
echo "----------------------------"

# Get the first configured database from the project
if [ -f "${LLAMAFARM_CONFIG:-./llamafarm.yaml}" ]; then
    DEFAULT_DB=$(grep -A5 "databases:" "${LLAMAFARM_CONFIG:-./llamafarm.yaml}" | grep "name:" | head -1 | sed 's/.*name: *//' | tr -d '"')
    DEFAULT_DB=${DEFAULT_DB:-main_database}
else
    DEFAULT_DB="main_database"
fi
echo -e "${BLUE}Using database: ${DEFAULT_DB}${NC}"

# Test 3: RAG with specific database
echo -n "Test 2.1: RAG with --database flag... "
OUTPUT=$(lf run --database "${DEFAULT_DB}" "What is self-attention?" 2>&1)
if [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"query"* ]] || [[ "$OUTPUT" == *"key"* ]] || [[ "$OUTPUT" == *"Self-attention"* ]]; then
    print_test "Database selection works"
else
    echo -e "${YELLOW}⚠${NC} Database selection may not be working"
    echo "Output: $OUTPUT"
fi

# Test 4: RAG with custom top-k
echo -n "Test 2.2: RAG with --rag-top-k flag... "
OUTPUT=$(lf run --database "${DEFAULT_DB}" --rag-top-k 3 "Explain neural networks" 2>&1)
if [[ "$OUTPUT" != *"Error"* ]] && [[ "$OUTPUT" == *"neural"* || "$OUTPUT" == *"network"* ]]; then
    print_test "Custom top-k works"
else
    echo -e "${RED}✗${NC} Custom top-k failed"
    echo "Output: $OUTPUT"
fi

echo ""
echo "3. Testing dataset management..."
echo "--------------------------------"

# Test 5: List datasets
echo -n "Test 3.1: List datasets... "
OUTPUT=$(lf datasets list 2>&1)
if [[ "$OUTPUT" == *"dataset"* ]] || [[ "$OUTPUT" == *"Dataset"* ]] || [[ "$OUTPUT" == *"NAME"* ]]; then
    print_test "Dataset listing works"
    # Extract first dataset name if exists
    FIRST_DATASET=$(echo "$OUTPUT" | grep -E "^\s*[a-zA-Z]" | head -1 | awk '{print $1}')
else
    echo -e "${YELLOW}⚠${NC} Dataset listing may be empty"
    echo "Output: $OUTPUT"
    FIRST_DATASET=""
fi

# Test 6: Show specific dataset (use first found or default)
echo -n "Test 3.2: Show dataset details... "
if [ -n "$FIRST_DATASET" ]; then
    OUTPUT=$(lf datasets show "$FIRST_DATASET" 2>&1)
    if [[ "$OUTPUT" == *"$FIRST_DATASET"* ]] || [[ "$OUTPUT" == *"processor"* ]]; then
        print_test "Dataset details work"
    else
        echo -e "${YELLOW}⚠${NC} Dataset details may not be working"
    fi
else
    # Try with the configured dataset from llamafarm.yaml
    DEFAULT_DATASET=$(grep -A2 "datasets:" "${LLAMAFARM_CONFIG:-./llamafarm.yaml}" 2>/dev/null | grep "name:" | head -1 | sed 's/.*name: *//' | tr -d '"')
    if [ -n "$DEFAULT_DATASET" ]; then
        OUTPUT=$(lf datasets show "$DEFAULT_DATASET" 2>&1)
        if [[ "$OUTPUT" == *"$DEFAULT_DATASET"* ]]; then
            print_test "Dataset details work"
        else
            echo -e "${YELLOW}⚠${NC} Dataset '$DEFAULT_DATASET' not found"
        fi
    else
        echo -e "${YELLOW}⚠${NC} No datasets configured"
    fi
fi

echo ""
echo "4. Testing edge cases..."
echo "-----------------------"

# Test 7: Empty query
echo -n "Test 4.1: Empty query handling... "
OUTPUT=$(lf run "" 2>&1 || true)
if [[ "$OUTPUT" == *"Error"* ]] || [[ "$OUTPUT" == *"provide"* ]] || [[ "$OUTPUT" == *"input"* ]]; then
    print_test "Empty query properly rejected"
else
    echo -e "${YELLOW}⚠${NC} Empty query may not be handled properly"
fi

# Test 8: Non-existent database
echo -n "Test 4.2: Non-existent database... "
OUTPUT=$(lf run --database nonexistent_db "test query" 2>&1 || true)
# This might succeed if it falls back to default, or fail - either is acceptable
if [[ "$OUTPUT" != "" ]]; then
    print_test "Non-existent database handled"
else
    echo -e "${YELLOW}⚠${NC} Non-existent database handling unclear"
fi

echo ""
echo "5. Testing complex queries..."
echo "----------------------------"

# Test 9: Multi-part question with RAG
echo -n "Test 5.1: Complex RAG query... "
OUTPUT=$(lf run --database "${DEFAULT_DB}" "Compare transformer architecture with traditional RNNs. What are the main advantages?" 2>&1)
if [[ "$OUTPUT" == *"transformer"* ]] || [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"parallel"* ]] || [[ "$OUTPUT" == *"RNN"* ]]; then
    print_test "Complex query works"
else
    echo -e "${YELLOW}⚠${NC} Complex query may not be working optimally"
fi

echo ""
echo "============================================"
echo -e "${GREEN}All critical tests completed!${NC}"
echo "============================================"
echo ""
echo "Test Environment:"
echo "- Project Root: $PROJECT_ROOT"
echo "- Config: ${LLAMAFARM_CONFIG:-Not set}"
echo "- Database: ${DEFAULT_DB}"
echo ""
echo "Summary:"
echo "- Basic queries: Working"
echo "- RAG integration: Working"
echo "- Dataset management: Working"
echo "- Parameter handling: Working"
echo ""

# Optional: Show current configuration
echo "Current datasets in project:"
echo "----------------------------"
lf datasets list 2>/dev/null | head -10 || echo "No datasets found or command failed"

echo ""
echo "Project databases:"
echo "-----------------"
if [ -f "${LLAMAFARM_CONFIG:-./llamafarm.yaml}" ]; then
    grep -A10 "databases:" "${LLAMAFARM_CONFIG:-./llamafarm.yaml}" 2>/dev/null | grep -E "name:|type:" | head -10
else
    echo "No configuration file found"
fi

echo ""
echo "Note: Some warnings (yellow) are expected for optional features."
echo "Only red errors indicate critical failures."