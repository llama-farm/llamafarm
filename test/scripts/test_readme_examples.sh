#!/bin/bash

# Test all examples from examples/README.md
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================"
echo "TESTING README EXAMPLES"
echo "================================"
echo ""

cd "$(dirname "$0")/.."

# Function to run test
run_test() {
    local name="$1"
    local cmd="$2"
    echo -n "Testing: $name... "
    if output=$(eval "$cmd" 2>&1); then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        echo -e "${RED}✗${NC}"
        echo "  Error: $output"
        return 1
    fi
}

echo "1. Dataset Management Tests"
echo "----------------------------"

# Create a test dataset
run_test "Create dataset" \
    "./lf datasets add readme-test -s universal_processor -b main_database"

# Add real example documents
run_test "Ingest research papers" \
    "./lf datasets ingest readme-test examples/rag_pipeline/sample_files/research_papers/*.txt"

# Add more document types
run_test "Ingest mixed documents" \
    "./lf datasets ingest readme-test examples/rag_pipeline/sample_files/code_documentation/*.md examples/rag_pipeline/sample_files/code/*.py"

# List datasets
run_test "List datasets" \
    "./lf datasets list | grep readme-test"

echo ""
echo "2. RAG Query Tests"
echo "------------------"

# Query without RAG
run_test "Basic query (no RAG)" \
    "./lf run 'What is 2+2?'"

# Query with RAG enabled
run_test "RAG query" \
    "./lf run --rag 'What is transformer architecture?'"

# Query specific database
run_test "RAG with database" \
    "./lf run --rag --database main_database 'Explain attention mechanism'"

# Customize retrieval parameters
run_test "RAG with top-k" \
    "./lf run --rag --database main_database --rag-top-k 10 'How do neural networks work?'"

# Use score threshold
run_test "RAG with threshold" \
    "./lf run --rag --database main_database --rag-score-threshold 0.7 'Best practices for API design'"

echo ""
echo "3. Working with Real Documents"
echo "-------------------------------"

# Create another test dataset
run_test "Create test-docs dataset" \
    "./lf datasets add test-docs -s universal_processor -b main_database"

# Ingest research papers
run_test "Ingest specific papers" \
    "./lf datasets ingest test-docs examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt examples/rag_pipeline/sample_files/research_papers/neural_scaling_laws.txt"

# Add code documentation
run_test "Ingest documentation" \
    "./lf datasets ingest test-docs examples/rag_pipeline/sample_files/code_documentation/api_reference.md examples/rag_pipeline/sample_files/code_documentation/implementation_guide.md"

# Query documents with RAG
run_test "Query transformer components" \
    "./lf run --rag --database main_database 'What are the key components of transformer architecture?'"

run_test "Query API methods" \
    "./lf run --rag --database main_database 'What are the API authentication methods?'"

run_test "Query scaling laws" \
    "./lf run --rag --database main_database 'Explain neural scaling laws'"

echo ""
echo "4. File Input Tests"
echo "-------------------"

# Test file input
echo "What is attention mechanism?" > /tmp/test_query.txt
run_test "File input with -f" \
    "./lf run -f /tmp/test_query.txt"
rm -f /tmp/test_query.txt

echo ""
echo "5. Complete Workflow Test"
echo "-------------------------"

# Create AI research dataset
run_test "Create ai-research dataset" \
    "./lf datasets add ai-research -s universal_processor -b main_database"

# Ingest all research papers
run_test "Ingest all papers" \
    "./lf datasets ingest ai-research examples/rag_pipeline/sample_files/research_papers/*.txt"

# Add documentation files
run_test "Add all documentation" \
    "./lf datasets ingest ai-research examples/rag_pipeline/sample_files/code_documentation/*.md"

# Verify ingestion
run_test "Verify dataset exists" \
    "./lf datasets list | grep ai-research"

# Test various RAG queries
run_test "Query transformer arch" \
    "./lf run --rag --database main_database 'What is transformer architecture?'"

run_test "Query attention in detail" \
    "./lf run --rag --database main_database 'Explain attention mechanism in transformers'"

run_test "Query API best practices" \
    "./lf run --rag --database main_database 'What are best practices for API development?'"

# Test with different parameters
run_test "Query with limited results" \
    "./lf run --rag --database main_database --rag-top-k 5 'Neural network scaling'"

run_test "Query with high threshold" \
    "./lf run --rag --database main_database --rag-score-threshold 0.8 'implementation details'"

echo ""
echo "6. Cleanup"
echo "----------"

# Clean up test datasets
run_test "Remove readme-test" \
    "./lf datasets remove readme-test"

run_test "Remove test-docs" \
    "./lf datasets remove test-docs"

run_test "Remove ai-research" \
    "./lf datasets remove ai-research"

echo ""
echo "================================"
echo -e "${GREEN}README EXAMPLES TEST COMPLETE${NC}"
echo "================================"
echo ""
echo "Summary:"
echo "✅ Dataset management commands work"
echo "✅ RAG queries work with various parameters"
echo "✅ Real document ingestion works"
echo "✅ File input works"
echo "✅ Complete workflow validated"
echo ""
echo "All examples from examples/README.md are functional!"