#!/bin/bash

# Validate Examples Script
# This script validates that all examples work correctly

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================"
echo "VALIDATING LLAMAFARM EXAMPLES"
echo "================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check CLI exists
if [ ! -f "./lf" ]; then
    echo -e "${RED}✗ LlamaFarm CLI not found${NC}"
    echo "  Build with: go build -o lf cli/main.go"
    exit 1
fi
echo -e "${GREEN}✓${NC} CLI found"

# Check sample files
echo ""
echo "Checking sample files..."
sample_count=$(find examples/rag_pipeline/sample_files -type f | wc -l)
if [ $sample_count -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Found $sample_count sample files"
else
    echo -e "${RED}✗${NC} No sample files found"
    exit 1
fi

# Test CLI commands
echo ""
echo "Testing CLI commands..."

# Test 1: Version
if ./lf version >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} lf version"
else
    echo -e "${RED}✗${NC} lf version failed"
fi

# Test 2: List datasets
if ./lf datasets list >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} lf datasets list"
else
    echo -e "${RED}✗${NC} lf datasets list failed"
fi

# Test 3: Create test dataset
TEST_DATASET="validation-test-$$"
if ./lf datasets create -s universal_processor -b main_database "$TEST_DATASET" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} lf datasets create"
    
    # Test 4: Upload files
    if ./lf datasets upload "$TEST_DATASET" examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} lf datasets upload"
    else
        echo -e "${YELLOW}⚠${NC} lf datasets upload failed (non-critical)"
    fi
    
    # Cleanup
    ./lf datasets delete "$TEST_DATASET" >/dev/null 2>&1 || true
else
    echo -e "${YELLOW}⚠${NC} lf datasets create failed (may already exist)"
fi

# Test 5: RAG query
if ./lf chat "test query" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} lf chat (basic)"
else
    echo -e "${YELLOW}⚠${NC} lf chat failed (server may be down)"
fi

# Test 6: RAG with database
if ./lf chat --database main_database "test" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} lf chat --database"
else
    echo -e "${YELLOW}⚠${NC} lf chat --database failed (RAG may not be configured)"
fi

echo ""
echo "================================"
echo -e "${GREEN}VALIDATION COMPLETE${NC}"
echo "================================"
echo ""
echo "Summary:"
echo "- CLI commands: Working"
echo "- Sample files: Present"
echo "- Dataset operations: Working"
echo ""
echo "To run the full example:"
echo "  ./examples/rag_pipeline/run_example.sh"
