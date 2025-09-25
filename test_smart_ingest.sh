#!/bin/bash

# Test script for the new smart ingest functionality
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Testing Smart Ingest Functionality${NC}"
echo "========================================"

# Test dataset name
TEST_DATASET="test_smart_ingest_$(date +%s)"
TEST_DB="test_smart_db"

echo -e "\n${YELLOW}Creating test dataset: ${TEST_DATASET}${NC}"
./lf datasets add "${TEST_DATASET}" -s universal_processor -b main_database

echo -e "\n${GREEN}Test 1: Ingest single file${NC}"
./lf datasets ingest "${TEST_DATASET}" examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt

echo -e "\n${GREEN}Test 2: Ingest multiple files${NC}"
./lf datasets ingest "${TEST_DATASET}" \
    examples/rag_pipeline/sample_files/research_papers/neural_scaling_laws.txt \
    examples/rag_pipeline/sample_files/research_papers/llm_scaling_laws.txt

echo -e "\n${GREEN}Test 3: Ingest with glob pattern (*.md)${NC}"
./lf datasets ingest "${TEST_DATASET}" "examples/rag_pipeline/sample_files/code_documentation/*.md"

echo -e "\n${GREEN}Test 4: Ingest entire directory${NC}"
./lf datasets ingest "${TEST_DATASET}" examples/rag_pipeline/sample_files/code/

echo -e "\n${GREEN}Test 5: Ingest directory recursively${NC}"
./lf datasets ingest "${TEST_DATASET}" examples/rag_pipeline/sample_files/ --recursive

echo -e "\n${GREEN}Test 6: Ingest with pattern filter${NC}"
./lf datasets ingest "${TEST_DATASET}" examples/rag_pipeline/sample_files/ --pattern "*.pdf" --recursive

echo -e "\n${GREEN}Test 7: Mixed input (files and directories)${NC}"
./lf datasets ingest "${TEST_DATASET}" \
    examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt \
    examples/rag_pipeline/sample_files/code_documentation/ \
    "examples/rag_pipeline/sample_files/fda/*.pdf"

echo -e "\n${GREEN}All tests completed!${NC}"
echo -e "${YELLOW}Dataset: ${TEST_DATASET}${NC}"
echo -e "\nTo clean up, run:"
echo "  ./lf datasets remove ${TEST_DATASET}"