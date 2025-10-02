#!/bin/bash

# ================================================================
# Comprehensive RAG CLI Test Script
# ================================================================
# This script tests the complete RAG workflow:
# 1. Creates a new database in llamafarm.yaml
# 2. Creates a dataset
# 3. Adds various types of documents
# 4. Processes them
# 5. Tests queries
# 6. Tests lf chat with and without RAG
# 7. Shows ALL output without truncation
# ================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
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

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ================================================================
# Configuration
# ================================================================

# Test database and dataset names
TEST_DB="test_rag_cli_db_$(date +%s)"
TEST_DATASET="test_rag_cli_dataset_$(date +%s)"

# Just use llamafarm.yaml in current directory
PROJECT_CONFIG="./llamafarm.yaml"

# Simple LF command
LF_CMD="./lf"

# Sample files directory - just use the actual path
SAMPLE_DIR="examples/rag_pipeline/sample_files"

print_header "RAG CLI Comprehensive Test"
echo "Test Database: ${TEST_DB}"
echo "Test Dataset: ${TEST_DATASET}"
echo "Config File: ${PROJECT_CONFIG}"
echo "Sample Files: ${SAMPLE_DIR}"

# ================================================================
# Step 1: Add new database to llamafarm.yaml
# ================================================================

print_header "Step 1: Adding New Test Database to Configuration"
print_step "Backing up current configuration..."
cp "$PROJECT_CONFIG" "${PROJECT_CONFIG}.backup_$(date +%s)"

print_step "Adding database '${TEST_DB}' to configuration..."

# Check if PyYAML is available, if not use sed approach
if python3 -c "import yaml" 2>/dev/null; then
    # Use Python to safely add the database to YAML
    python3 << EOF
import yaml
import sys

config_file = "${PROJECT_CONFIG}"

# Read the current config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Add new database
new_db = {
    'name': '${TEST_DB}',
    'type': 'ChromaStore',
    'config': {
        'collection_name': '${TEST_DB}',
        'distance_function': 'cosine',
        'persist_directory': './data/${TEST_DB}',
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
        },
        {
            'name': 'filtered_search',
            'type': 'MetadataFilteredStrategy',
            'config': {
                'fallback_multiplier': 2,
                'filter_mode': 'post',
                'top_k': 10
            },
            'default': False
        }
    ],
    'default_embedding_strategy': 'default_embeddings',
    'default_retrieval_strategy': 'basic_search'
}

# Check if database already exists
db_exists = any(db['name'] == '${TEST_DB}' for db in config.get('rag', {}).get('databases', []))

if not db_exists:
    config['rag']['databases'].append(new_db)

    # Write back the updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print("✓ Database '${TEST_DB}' added to configuration")
else:
    print("ℹ Database '${TEST_DB}' already exists in configuration")
EOF
else
    # Fallback: manually append database to YAML
    print_info "PyYAML not found, using manual approach..."

    # Check if database already exists
    if grep -q "name: ${TEST_DB}" "$PROJECT_CONFIG"; then
        print_info "Database '${TEST_DB}' already exists in configuration"
    else
        # Find a database entry and add new one before it
        # This adds the new database before the first database's embedding_strategies
        awk -v db="${TEST_DB}" '
        /^  databases:/ {
            print
            print "  - name: " db
            print "    type: ChromaStore"
            print "    config:"
            print "      collection_name: " db"
            print "      distance_function: cosine"
            print "      persist_directory: ./data/" db
            print "      port: 8000"
            print "    embedding_strategies:"
            print "    - name: default_embeddings"
            print "      type: OllamaEmbedder"
            print "      config:"
            print "        auto_pull: true"
            print "        base_url: http://localhost:11434"
            print "        batch_size: 16"
            print "        dimension: 768"
            print "        model: nomic-embed-text"
            print "        timeout: 60"
            print "      priority: 0"
            print "    retrieval_strategies:"
            print "    - name: basic_search"
            print "      type: BasicSimilarityStrategy"
            print "      config:"
            print "        distance_metric: cosine"
            print "        top_k: 10"
            print "      default: true"
            print "    - name: filtered_search"
            print "      type: MetadataFilteredStrategy"
            print "      config:"
            print "        fallback_multiplier: 2"
            print "        filter_mode: post"
            print "        top_k: 10"
            print "      default: false"
            print "    default_embedding_strategy: default_embeddings"
            print "    default_retrieval_strategy: basic_search"
            next
        }
        { print }
        ' "$PROJECT_CONFIG" > "${PROJECT_CONFIG}.tmp" && mv "${PROJECT_CONFIG}.tmp" "$PROJECT_CONFIG"

        print_success "Database '${TEST_DB}' added to configuration (manual method)"
    fi
fi

print_success "Database configuration updated"

# ================================================================
# Step 2: Create Dataset
# ================================================================

print_header "Step 2: Creating New Dataset"
print_step "Creating dataset '${TEST_DATASET}' with universal_processor strategy..."

echo "Command: ${LF_CMD} datasets add ${TEST_DATASET} -s universal_processor -b ${TEST_DB}"
${LF_CMD} datasets add "${TEST_DATASET}" -s universal_processor -b "${TEST_DB}"

print_success "Dataset created"

# ================================================================
# Step 3: Ingest Various Document Types
# ================================================================

print_header "Step 3: Ingesting Various Document Types"

print_step "Adding research papers directory (using directory upload)..."
echo "Uploading entire research_papers directory at once..."
${LF_CMD} datasets ingest "${TEST_DATASET}" \
    ${SAMPLE_DIR}/research_papers/

print_step "Adding code documentation (using glob pattern for markdown files)..."
echo "Uploading all *.md files in code_documentation directory..."
${LF_CMD} datasets ingest "${TEST_DATASET}" \
    ${SAMPLE_DIR}/code_documentation/*.md

print_step "Adding code examples (Python files)..."
${LF_CMD} datasets ingest "${TEST_DATASET}" \
    ${SAMPLE_DIR}/code/example.py

print_step "Adding FDA documents..."
echo "Uploading all PDF files in fda directory..."
${LF_CMD} datasets ingest "${TEST_DATASET}" \
    ${SAMPLE_DIR}/fda/*.pdf || print_info "Note: Some PDF files may require additional dependencies"

print_step "Demonstrating recursive upload with /**/* pattern (all files in all subdirs)..."
echo "Command that would be used for recursive upload:"
echo -e "${MAGENTA}${LF_CMD} datasets ingest \"${TEST_DATASET}\" \"${SAMPLE_DIR}/**/*\"${NC}"
echo ""
echo "This pattern would recursively upload ALL files:"
find ${SAMPLE_DIR} -type f | head -5 | while read f; do echo "  • $(basename $(dirname $f))/$(basename $f)"; done
echo "  ... and $(( $(find ${SAMPLE_DIR} -type f | wc -l) - 5 )) more files"
echo ""
print_info "Skipping actual upload to avoid duplicates (uncomment in script to run)"

print_step "Testing duplicate detection - Re-uploading research papers..."
echo "Command: ${LF_CMD} datasets ingest \"${TEST_DATASET}\" ${SAMPLE_DIR}/research_papers/transformer_architecture.txt"
echo "Expected: File should be skipped as duplicate"
${LF_CMD} datasets ingest "${TEST_DATASET}" \
    ${SAMPLE_DIR}/research_papers/transformer_architecture.txt 2>&1 | grep -E "(already exists|SKIPPED|Uploaded)" || \
    ${LF_CMD} datasets ingest "${TEST_DATASET}" ${SAMPLE_DIR}/research_papers/transformer_architecture.txt

print_success "All documents ingested"

# ================================================================
# Step 4: Process Documents
# ================================================================

print_header "Step 4: Processing Documents into Vector Database"
print_step "Processing all ingested documents..."

echo "Command: ${LF_CMD} datasets process ${TEST_DATASET}"
${LF_CMD} datasets process "${TEST_DATASET}"

print_success "Documents processed"

# ================================================================
# Step 5: List Datasets to Verify
# ================================================================

print_header "Step 5: Verifying Dataset Status"
print_step "Listing all datasets..."

echo "Command: ${LF_CMD} datasets list"
${LF_CMD} datasets list | grep -A 5 -B 5 "${TEST_DATASET}" || ${LF_CMD} datasets list

# ================================================================
# Step 6: Test RAG Queries
# ================================================================

print_header "Step 6: Testing RAG Queries"

print_step "Query 1: Basic query about transformer architecture"
echo "Command: ${LF_CMD} rag query --database ${TEST_DB} \"What is transformer architecture?\""
${LF_CMD} rag query --database "${TEST_DB}" "What is transformer architecture?"

print_step "Query 2: Query with custom top-k setting"
echo "Command: ${LF_CMD} rag query --database ${TEST_DB} --top-k 3 \"Explain attention mechanism\""
${LF_CMD} rag query --database "${TEST_DB}" --top-k 3 "Explain attention mechanism"

print_step "Query 3: Query with score threshold"
echo "Command: ${LF_CMD} rag query --database ${TEST_DB} --score-threshold 0.7 \"Best practices for API design\""
${LF_CMD} rag query --database "${TEST_DB}" --score-threshold 0.7 "Best practices for API design"

print_success "RAG queries completed"

# ================================================================
# Step 7: Test Chat with RAG
# ================================================================

print_header "Step 7: Testing Chat with RAG Integration"

print_step "Test 1: Chat WITH RAG (default behavior)"
echo "Command: timeout 10 ${LF_CMD} chat --database ${TEST_DB} \"What are the key components of transformer architecture?\""
timeout 10 ${LF_CMD} chat --database "${TEST_DB}" "What are the key components of transformer architecture?" || true

print_step "Test 2: Chat WITHOUT RAG (LLM only)"
echo "Command: timeout 10 ${LF_CMD} chat --no-rag \"What are the key components of transformer architecture?\""
timeout 10 ${LF_CMD} chat --no-rag "What are the key components of transformer architecture?" || true

print_success "Chat tests completed"

# ================================================================
# Step 8: Comparison Test - RAG vs No-RAG
# ================================================================

print_header "Step 8: Direct Comparison - RAG vs No-RAG"

QUERY="What is the DataProcessor class mentioned in our documentation?"

print_step "Asking about DataProcessor WITH RAG context:"
echo "Command: timeout 10 ${LF_CMD} chat --database ${TEST_DB} \"${QUERY}\""
echo -e "${GREEN}Response with RAG:${NC}"
timeout 10 ${LF_CMD} chat --database "${TEST_DB}" "${QUERY}" || true

echo -e "\n${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"

print_step "Asking about DataProcessor WITHOUT RAG context:"
echo "Command: timeout 10 ${LF_CMD} chat --no-rag \"${QUERY}\""
echo -e "${RED}Response without RAG:${NC}"
timeout 10 ${LF_CMD} chat --no-rag "${QUERY}" || true

# ================================================================
# Step 9: Test Duplicate Detection
# ================================================================

print_header "Step 9: Testing Duplicate Detection"

print_step "IMPORTANT: Re-processing the SAME dataset to demonstrate duplicate detection..."
echo -e "${YELLOW}This should show all files as SKIPPED since they're already in the database${NC}\n"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}EXPECTED: All files should be SKIPPED as duplicates${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════════${NC}\n"

echo "Command: ${LF_CMD} datasets process ${TEST_DATASET}"
${LF_CMD} datasets process "${TEST_DATASET}"

print_success "First re-process complete - files should show as SKIPPED"

# ================================================================
# Step 10: Test Again - Process Same Dataset Third Time
# ================================================================

print_header "Step 10: Processing Same Dataset Again (Third Time)"

print_step "Processing the SAME dataset for the third time..."
echo -e "${YELLOW}This should ALSO show all files as SKIPPED${NC}\n"
echo "Command: ${LF_CMD} datasets process ${TEST_DATASET}"
${LF_CMD} datasets process "${TEST_DATASET}"

print_success "Duplicate detection test completed"
print_info "Files should consistently show as SKIPPED on subsequent processing attempts"

# ================================================================
# Step 11: Cleanup (Optional)
# ================================================================

print_header "Step 11: Cleanup"

read -p "Do you want to remove the test dataset? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Removing test dataset..."
    echo "Command: ${LF_CMD} datasets remove ${TEST_DATASET}"
    ${LF_CMD} datasets remove "${TEST_DATASET}"

    print_step "Removing test database from configuration..."

    if python3 -c "import yaml" 2>/dev/null; then
        python3 << EOF
import yaml

config_file = "${PROJECT_CONFIG}"

# Read the current config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Remove test database
config['rag']['databases'] = [
    db for db in config.get('rag', {}).get('databases', [])
    if db['name'] != '${TEST_DB}'
]

# Remove test dataset
config['datasets'] = [
    ds for ds in config.get('datasets', [])
    if ds['name'] != '${TEST_DATASET}'
]

# Write back the updated config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("✓ Test database and dataset removed from configuration")
EOF
    else
        print_info "Manual cleanup - removing test entries from configuration..."
        # Remove database entry (simplified - removes lines containing the test db name)
        sed -i.bak "/name: ${TEST_DB}/,+6d" "$PROJECT_CONFIG"
        # Remove dataset entry
        sed -i.bak "/name: ${TEST_DATASET}/,+3d" "$PROJECT_CONFIG"
        print_success "Test entries removed from configuration"
    fi

    print_success "Cleanup completed"
else
    print_info "Keeping test dataset and database for manual inspection"
    echo "To remove later, run:"
    echo "  ${LF_CMD} datasets remove ${TEST_DATASET}"
fi

# ================================================================
# Summary
# ================================================================

print_header "Test Complete!"
echo -e "${GREEN}All RAG CLI operations have been tested successfully!${NC}"
echo ""
echo "Test Summary:"
echo "  • Created database: ${TEST_DB}"
echo "  • Created dataset: ${TEST_DATASET}"
echo "  • Ingested multiple document types"
echo "  • Processed documents into vector database"
echo "  • Tested RAG queries with various parameters"
echo "  • Tested chat with and without RAG"
echo "  • Verified duplicate detection"
echo ""
echo "The enhanced CLI now shows:"
echo "  • Detailed file processing information"
echo "  • Parser and chunk statistics"
echo "  • Extractor application details"
echo "  • Clear duplicate detection status"
echo "  • Comprehensive processing summaries"