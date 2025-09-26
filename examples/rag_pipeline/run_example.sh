#!/bin/bash

# RAG Pipeline CLI Example
# This script demonstrates the complete RAG workflow using the LlamaFarm CLI

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "============================================"
echo "RAG PIPELINE CLI EXAMPLE"
echo "============================================"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root for CLI commands
cd "$PROJECT_ROOT"

# Check if lf CLI exists
if [ ! -f "./lf" ]; then
    echo -e "${RED}Error: LlamaFarm CLI not found!${NC}"
    echo "Please build it first: go build -o lf cli/main.go"
    exit 1
fi

echo -e "${BLUE}📋 Step 1: Check CLI version${NC}"
./lf version
echo ""

echo -e "${BLUE}📋 Step 2: List existing datasets${NC}"
./lf datasets list
echo ""

# Dataset name for this example
DATASET_NAME="rag-example-dataset"

echo -e "${BLUE}📋 Step 3: Create example dataset${NC}"
# Remove if exists
./lf datasets remove "$DATASET_NAME" 2>/dev/null || true
./lf datasets add "$DATASET_NAME" -s universal_processor -b main_database
echo ""

echo -e "${BLUE}📋 Step 4: Ingest sample documents${NC}"
echo "Adding research papers..."
./lf datasets ingest "$DATASET_NAME" examples/rag_pipeline/sample_files/research_papers/*.txt

echo "Adding code documentation..."
./lf datasets ingest "$DATASET_NAME" examples/rag_pipeline/sample_files/code_documentation/*.md

echo "Adding example code..."
./lf datasets ingest "$DATASET_NAME" examples/rag_pipeline/sample_files/code/*.py
echo ""

echo -e "${BLUE}📋 Step 5: Verify dataset${NC}"
./lf datasets show "$DATASET_NAME"
echo ""

echo -e "${BLUE}📋 Step 6: Test queries${NC}"
echo ""

echo -e "${YELLOW}Query 1: Without RAG (baseline)${NC}"
echo "Question: What is transformer architecture?"
./lf chat "What is transformer architecture? Give a brief answer."
echo ""

echo -e "${YELLOW}Query 2: With RAG enabled${NC}"
echo "Question: What is transformer architecture?"
./lf chat --rag --database main_database "What is transformer architecture? Give a brief answer with specific details."
echo ""

echo -e "${YELLOW}Query 3: Code-related query with RAG${NC}"
echo "Question: What is the DataProcessor class?"
./lf chat --rag --database main_database "What is the DataProcessor class and what methods does it have?"
echo ""

echo -e "${YELLOW}Query 4: Research paper query with custom top-k${NC}"
echo "Question: Explain neural scaling laws"
./lf chat --rag --database main_database --rag-top-k 5 "Explain neural scaling laws and their implications"
echo ""

echo "============================================"
echo -e "${GREEN}✅ RAG Pipeline Example Complete!${NC}"
echo "============================================"
echo ""
echo "Summary:"
echo "- Created dataset: $DATASET_NAME"
echo "- Ingested multiple document types (txt, md, py)"
echo "- Demonstrated queries with and without RAG"
echo "- Showed custom retrieval parameters"
echo ""
echo "To clean up, run:"
echo "  ./lf datasets remove $DATASET_NAME"
echo ""
echo "To explore more:"
echo "  ./lf chat --help        # See all query options"
echo "  ./lf datasets --help   # See dataset management options"
echo "  ./lf rag --help       # See RAG-specific commands"