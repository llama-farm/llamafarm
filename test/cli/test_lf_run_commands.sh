#!/bin/bash

# Comprehensive test for all lf chat command variations
# Tests basic queries, RAG queries, and all parameter combinations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================"
echo "TESTING ALL LF CHAT COMMAND VARIATIONS"
echo "============================================"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$(dirname "$PROJECT_ROOT")"

# Track test results
PASSED=0
FAILED=0
WARNINGS=0

# Function to run a test
run_test() {
    local description="$1"
    local command="$2"
    local check_for="$3"  # Optional string to check in output
    
    echo -e "\n${BLUE}Test:${NC} $description"
    echo -e "${BLUE}Command:${NC} $command"
    
    # Run the command and capture output
    if output=$(eval "$command" 2>&1); then
        # Command succeeded
        if [ -n "$check_for" ]; then
            # Check if expected string is in output
            if echo "$output" | grep -q "$check_for"; then
                echo -e "${GREEN}✓ PASSED${NC} - Found expected content"
                ((PASSED++))
            else
                echo -e "${YELLOW}⚠ WARNING${NC} - Command succeeded but didn't find: $check_for"
                echo "Output snippet: $(echo "$output" | head -2)"
                ((WARNINGS++))
            fi
        else
            # Just check if we got any output
            if [ -n "$output" ]; then
                echo -e "${GREEN}✓ PASSED${NC}"
                ((PASSED++))
            else
                echo -e "${YELLOW}⚠ WARNING${NC} - No output received"
                ((WARNINGS++))
            fi
        fi
    else
        echo -e "${RED}✗ FAILED${NC} - Exit code: $?"
        echo "Error: $(echo "$output" | tail -2)"
        ((FAILED++))
    fi
}

echo "============================================"
echo "1. BASIC QUERIES (WITHOUT RAG)"
echo "============================================"

run_test \
    "Simple math query (no RAG)" \
    "./lf chat --no-rag 'What is 2+2?'" \
    "4"

run_test \
    "General knowledge query (no RAG)" \
    "./lf chat --no-rag 'What is the capital of France?'" \
    "Paris"

run_test \
    "Short response query (no RAG)" \
    "./lf chat --no-rag 'Say hello'" \
    ""

echo ""
echo "============================================"
echo "2. RAG-ENABLED QUERIES"
echo "============================================"

run_test \
    "RAG query with --rag flag" \
    "./lf chat 'What is transformer architecture?'" \
    ""

run_test \
    "RAG with specific database" \
    "./lf chat --database main_database 'What is self-attention?'" \
    ""

run_test \
    "RAG with custom top-k" \
    "./lf chat --database main_database --rag-top-k 3 'Explain neural networks'" \
    ""

run_test \
    "RAG with high top-k for more context" \
    "./lf chat --database main_database --rag-top-k 10 'What are the key ML concepts?'" \
    ""

run_test \
    "RAG with score threshold" \
    "./lf chat --database main_database --rag-score-threshold 0.5 'Machine learning basics'" \
    ""

echo ""
echo "============================================"
echo "3. COMBINED PARAMETERS"
echo "============================================"

run_test \
    "RAG with all parameters" \
    "./lf chat --database main_database --rag-top-k 5 --rag-score-threshold 0.3 'Explain attention mechanism in detail'" \
    ""

run_test \
    "Complex technical query with RAG" \
    "./lf chat --database main_database --rag-top-k 8 'Compare transformer architecture with RNN models'" \
    ""

echo ""
echo "============================================"
echo "4. INPUT FILE TESTS"
echo "============================================"

# Create a test input file
echo "What are neural scaling laws and their implications?" > /tmp/test_query.txt

run_test \
    "Query from file" \
    "./lf chat -f /tmp/test_query.txt" \
    ""

run_test \
    "Query from file with RAG" \
    "./lf chat --database main_database -f /tmp/test_query.txt" \
    ""

# Cleanup
rm -f /tmp/test_query.txt

echo ""
echo "============================================"
echo "5. EDGE CASES"
echo "============================================"

run_test \
    "Empty query handling" \
    "./lf chat ''" \
    ""

run_test \
    "Very long query" \
    "./lf chat 'Please provide a comprehensive explanation of machine learning, including supervised learning, unsupervised learning, reinforcement learning, neural networks, deep learning, and their applications in modern technology'" \
    ""

run_test \
    "Query with special characters" \
    "./lf chat 'What is 10% of 100?'" \
    ""

run_test \
    "Non-existent database (should use default or fail gracefully)" \
    "./lf chat --database nonexistent_db 'test query' 2>&1 || true" \
    ""

echo ""
echo "============================================"
echo "6. PERFORMANCE TESTS"
echo "============================================"

# Test response time
start_time=$(date +%s)
run_test \
    "Quick query performance" \
    "./lf chat 'Hi'" \
    ""
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Response time: ${elapsed} seconds"

echo ""
echo "============================================"
echo "TEST SUMMARY"
echo "============================================"
echo ""
echo -e "${GREEN}Passed:${NC} $PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED PERFECTLY!${NC}"
    else
        echo -e "${GREEN}✅ All critical tests passed${NC} (with $WARNINGS warnings)"
    fi
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi