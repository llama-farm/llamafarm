#!/bin/bash

# Quick test for lf run commands
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================"
echo "QUICK LF RUN TEST"
echo "================================"
echo ""

cd "$(dirname "$0")/.."

PASSED=0
FAILED=0

test_command() {
    local desc="$1"
    local cmd="$2"
    echo -e "\n${BLUE}Testing:${NC} $desc"
    if output=$($cmd 2>&1); then
        if [ -n "$output" ]; then
            echo -e "${GREEN}✓ PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${YELLOW}⚠ No output${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
}

# Basic queries
test_command "Simple math" "./lf run 'What is 2+2?'"
test_command "Hello query" "./lf run 'Say hello'"
test_command "General knowledge" "./lf run 'What is the capital of France?'"

# RAG queries
test_command "RAG basic" "./lf run --rag 'What is transformer architecture?'"
test_command "RAG with database" "./lf run --rag --database main_database 'What is attention?'"
test_command "RAG with top-k" "./lf run --rag --database main_database --rag-top-k 3 'Neural networks'"
test_command "RAG with threshold" "./lf run --rag --database main_database --rag-score-threshold 0.5 'Machine learning'"

# Combined parameters
test_command "RAG all params" "./lf run --rag --database main_database --rag-top-k 5 --rag-score-threshold 0.3 'Deep learning'"

# File input
echo "Test query from file" > /tmp/test_query.txt
test_command "File input" "./lf run -f /tmp/test_query.txt"
test_command "File with RAG" "./lf run --rag --database main_database -f /tmp/test_query.txt"
rm -f /tmp/test_query.txt

echo ""
echo "================================"
echo "SUMMARY"
echo "================================"
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${RED}Failed:${NC} $FAILED"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED!${NC}"
else
    echo -e "\n${YELLOW}⚠ Some tests failed${NC}"
fi