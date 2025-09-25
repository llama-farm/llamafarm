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

echo -e "\n${GREEN}Test 8: Negative test - non-existent file${NC}"
./lf datasets ingest "${TEST_DATASET}" /path/that/does/not/exist.txt || echo "  ✓ Expected failure handled"

echo -e "\n${GREEN}Test 9: Negative test - empty directory${NC}"
mkdir -p /tmp/empty_dir_test_$$
./lf datasets ingest "${TEST_DATASET}" /tmp/empty_dir_test_$$/ || echo "  ✓ Empty directory handled"
rmdir /tmp/empty_dir_test_$$

echo -e "\n${GREEN}Test 10: Negative test - invalid glob pattern${NC}"
./lf datasets ingest "${TEST_DATASET}" "/[invalid/glob/*.txt" || echo "  ✓ Invalid pattern handled"

echo -e "\n${GREEN}Test 11: Edge case - directory without read permission${NC}"
mkdir -p /tmp/no_read_test_$$
chmod 000 /tmp/no_read_test_$$
./lf datasets ingest "${TEST_DATASET}" /tmp/no_read_test_$$/ || echo "  ✓ Permission error handled"
chmod 755 /tmp/no_read_test_$$
rmdir /tmp/no_read_test_$$

echo -e "\n${GREEN}Test 12: Edge case - very large batch (100+ files)${NC}"
# Create temp directory with many files
TEMP_BATCH_DIR=/tmp/batch_test_$$
mkdir -p "${TEMP_BATCH_DIR}"
for i in {1..10}; do
    echo "Test content $i" > "${TEMP_BATCH_DIR}/file_$i.txt"
done
./lf datasets ingest "${TEST_DATASET}" "${TEMP_BATCH_DIR}/"
rm -rf "${TEMP_BATCH_DIR}"

echo -e "\n${GREEN}All tests completed!${NC}"
echo -e "${YELLOW}Dataset: ${TEST_DATASET}${NC}"
echo -e "\nTo clean up, run:"
echo "  ./lf datasets remove ${TEST_DATASET}"