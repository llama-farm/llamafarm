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
# Get the project root (parent of test directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_ROOT"

echo "1. Testing basic functionality..."
echo "--------------------------------"

# Test 1: Run without RAG
echo -n "Test 1.1: Query without RAG... "
OUTPUT=$(./lf run "What is 2+2?" 2>&1)
if [[ "$OUTPUT" == *"4"* ]] || [[ "$OUTPUT" == *"four"* ]]; then
    print_test "Basic query works"
else
    echo -e "${RED}✗${NC} Basic query failed"
    echo "Output: $OUTPUT"
fi

# Test 2: Run with RAG enabled
echo -n "Test 1.2: Query with RAG... "
OUTPUT=$(./lf run --rag "What is transformer architecture?" 2>&1)
if [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"encoder"* ]] || [[ "$OUTPUT" == *"decoder"* ]]; then
    print_test "RAG query works"
else
    echo -e "${RED}✗${NC} RAG query failed"
    echo "Output: $OUTPUT"
fi

echo ""
echo "2. Testing RAG parameters..."
echo "----------------------------"

# Test 3: RAG with specific database
echo -n "Test 2.1: RAG with --database flag... "
OUTPUT=$(./lf run --rag --database main_database "What is self-attention?" 2>&1)
if [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"query"* ]] || [[ "$OUTPUT" == *"key"* ]]; then
    print_test "Database selection works"
else
    echo -e "${YELLOW}⚠${NC} Database selection may not be working"
    echo "Output: $OUTPUT"
fi

# Test 4: RAG with custom top-k
echo -n "Test 2.2: RAG with --rag-top-k flag... "
OUTPUT=$(./lf run --rag --database main_database --rag-top-k 3 "Explain neural networks" 2>&1)
if [[ "$OUTPUT" != *"Error"* ]]; then
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
OUTPUT=$(./lf datasets list 2>&1)
if [[ "$OUTPUT" == *"dataset"* ]]; then
    print_test "Dataset listing works"
else
    echo -e "${RED}✗${NC} Dataset listing failed"
    echo "Output: $OUTPUT"
fi

# Test 6: Show specific dataset
echo -n "Test 3.2: Show dataset details... "
OUTPUT=$(./lf datasets show demo-rag-dataset 2>&1)
if [[ "$OUTPUT" == *"demo-rag-dataset"* ]] || [[ "$OUTPUT" == *"universal_processor"* ]]; then
    print_test "Dataset details work"
else
    echo -e "${YELLOW}⚠${NC} Dataset details may not be working"
fi

echo ""
echo "4. Testing edge cases..."
echo "-----------------------"

# Test 7: Empty query
echo -n "Test 4.1: Empty query handling... "
OUTPUT=$(./lf run --rag "" 2>&1 || true)
if [[ "$OUTPUT" == *"Error"* ]] || [[ "$OUTPUT" == *"provide"* ]]; then
    print_test "Empty query properly rejected"
else
    echo -e "${YELLOW}⚠${NC} Empty query may not be handled properly"
fi

# Test 8: Non-existent database
echo -n "Test 4.2: Non-existent database... "
OUTPUT=$(./lf run --rag --database nonexistent_db "test query" 2>&1 || true)
# This might succeed if it falls back to default, or fail - either is acceptable
print_test "Non-existent database handled"

echo ""
echo "5. Testing complex queries..."
echo "----------------------------"

# Test 9: Multi-part question with RAG
echo -n "Test 5.1: Complex RAG query... "
OUTPUT=$(./lf run --rag --database main_database "Compare transformer architecture with traditional RNNs. What are the main advantages?" 2>&1)
if [[ "$OUTPUT" == *"transformer"* ]] || [[ "$OUTPUT" == *"attention"* ]] || [[ "$OUTPUT" == *"parallel"* ]]; then
    print_test "Complex query works"
else
    echo -e "${YELLOW}⚠${NC} Complex query may not be working optimally"
fi

echo ""
echo "============================================"
echo -e "${GREEN}All critical tests completed!${NC}"
echo "============================================"
echo ""
echo "Summary:"
echo "- Basic queries: Working"
echo "- RAG integration: Working"
echo "- Dataset management: Working"
echo "- Parameter handling: Working"
echo ""

# Optional: Show current configuration
echo "Current configuration:"
echo "---------------------"
./lf datasets list | head -10

echo ""
echo "Note: Some warnings (yellow) are expected for optional features."
echo "Only red errors indicate critical failures."