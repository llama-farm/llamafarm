#!/bin/bash

# ================================================================
# Parser Chunking Verification Test Script
# ================================================================
# This script tests that each parser is actually chunking documents
# by creating a fresh database and dataset for each test
# ================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored headers
print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "\n${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# ================================================================
# Configuration
# ================================================================

PROJECT_CONFIG="./llamafarm.yaml"
LF_CMD="./lf"
SAMPLE_DIR="examples/rag_pipeline/sample_files"

# ================================================================
# Helper Functions
# ================================================================

add_database() {
    local DB_NAME=$1
    print_step "Adding database '${DB_NAME}' to configuration..."

    python3 << EOF
import yaml

config_file = "${PROJECT_CONFIG}"

# Read the current config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Add new database
new_db = {
    'name': '${DB_NAME}',
    'type': 'ChromaStore',
    'config': {
        'collection_name': '${DB_NAME}',
        'distance_function': 'cosine',
        'persist_directory': './data/${DB_NAME}',
        'port': 8000
    },
    'embedding_strategies': [
        {
            'name': 'default_embeddings',
            'type': 'OllamaEmbedder',
            'config': {
                'auto_pull': True,
                'base_url': 'http://localhost:11434',
                'batch_size': 16,
                'dimension': 768,
                'model': 'nomic-embed-text',
                'timeout': 60
            },
            'priority': 0
        }
    ],
    'retrieval_strategies': [
        {
            'name': 'basic_search',
            'type': 'BasicSimilarityStrategy',
            'config': {
                'distance_metric': 'cosine',
                'top_k': 10
            },
            'default': True
        }
    ],
    'default_embedding_strategy': 'default_embeddings',
    'default_retrieval_strategy': 'basic_search'
}

# Check if database already exists
db_exists = any(db['name'] == '${DB_NAME}' for db in config.get('rag', {}).get('databases', []))

if not db_exists:
    config['rag']['databases'].append(new_db)
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("✓ Database added")
else:
    print("ℹ Database already exists")
EOF
}

test_parser() {
    local PARSER_NAME=$1
    local FILE_PATH=$2
    local EXPECTED_CHUNKS=$3

    print_header "Testing ${PARSER_NAME}"

    # Create unique names (lowercase with underscores only)
    local TEST_DB="test_$(echo ${PARSER_NAME} | tr '[:upper:]' '[:lower:]')_$(date +%s)"
    local TEST_DATASET="dataset_$(echo ${PARSER_NAME} | tr '[:upper:]' '[:lower:]')_$(date +%s)"

    # Add database
    add_database "${TEST_DB}"

    # Create dataset
    print_step "Creating dataset..."
    ${LF_CMD} datasets add "${TEST_DATASET}" -s universal_processor -b "${TEST_DB}"

    # Ingest file
    print_step "Ingesting file: ${FILE_PATH}"
    ${LF_CMD} datasets ingest "${TEST_DATASET}" "${FILE_PATH}"

    # Process
    print_step "Processing file (watch for chunk count)..."
    ${LF_CMD} datasets process "${TEST_DATASET}" 2>&1 | tee /tmp/parser_test_output.log

    # Extract chunk info from output
    echo -e "\n${BLUE}Checking for chunking evidence...${NC}"
    grep -i "chunks\|chunk_index\|total_chunks" /tmp/parser_test_output.log || echo "No chunk information found in output"

    print_success "${PARSER_NAME} test complete"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"
}

# ================================================================
# Main Tests
# ================================================================

print_header "Parser Chunking Verification Tests"

# Test 1: PDF Parser (PyPDF2 - priority 10)
test_parser "PDFParser_PyPDF2" \
    "${SAMPLE_DIR}/fda/761248_2024_Orig1s000OtherActionLtrs.pdf" \
    "Multiple chunks expected"

# Test 2: PDF Parser (LlamaIndex - priority 50, should not be used due to PyPDF2)
# We'll test this by temporarily changing priority

# Test 3: Markdown Parser (Python - priority 100)
test_parser "MarkdownParser_Python" \
    "${SAMPLE_DIR}/code_documentation/implementation_guide.md" \
    "Multiple chunks expected (sections)"

# Test 4: Text Parser (Python - priority 100)
test_parser "TextParser_Python" \
    "${SAMPLE_DIR}/research_papers/neural_scaling_laws.txt" \
    "Multiple chunks expected (sentences)"

print_header "All Parser Tests Complete!"
echo -e "${GREEN}Check the output above for chunk counts in each parser test${NC}"
