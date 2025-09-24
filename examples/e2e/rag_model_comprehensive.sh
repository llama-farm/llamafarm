#!/bin/bash

# ================================================================
# Comprehensive Multi-Model & Multi-Database RAG Demo
# ================================================================
# This demo showcases LlamaFarm's capabilities for:
# 1. Multiple named model configurations (primary, creative, precise)
# 2. Multiple RAG databases for different use cases
# 3. Model selection for different tasks
# 4. Database selection based on content type
# 5. Advanced querying with different strategies
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

print_demo() {
    echo -e "${MAGENTA}🎭 DEMO: $1${NC}"
}

# ================================================================
# Configuration
# ================================================================

# Database names for different use cases
TECHNICAL_DB="technical_knowledge_db"
CREATIVE_DB="creative_content_db"
RESEARCH_DB="research_papers_db"

# Dataset names
TECHNICAL_DATASET="technical_docs_$(date +%s)"
CREATIVE_DATASET="creative_content_$(date +%s)"
RESEARCH_DATASET="research_papers_$(date +%s)"

# Project configuration path
PROJECT_CONFIG="$HOME/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml"
PROJECT_CONFIG_DIR="$(dirname "$PROJECT_CONFIG")"

# LlamaFarm CLI command
LF_PATH=${LF_PATH:-"./lf"}
LF_CMD="${LF_PATH} --cwd $PROJECT_CONFIG_DIR"

# Sample files directory
SAMPLE_DIR="${SAMPLE_DIR:-examples/rag_pipeline/sample_files}"

print_header "🚀 LlamaFarm Multi-Model & Multi-Database RAG Demo"
echo "This demo showcases:"
echo "  • Multiple named model configurations"
echo "  • Multiple specialized RAG databases"
echo "  • Intelligent model selection for different tasks"
echo "  • Database routing based on content type"
echo ""
echo "Databases to create:"
echo "  • ${TECHNICAL_DB} - For code and technical documentation"
echo "  • ${CREATIVE_DB} - For creative writing and articles"
echo "  • ${RESEARCH_DB} - For research papers and academic content"

# ================================================================
# Part 1: Model Management Demo
# ================================================================

print_header "Part 1: Model Management Capabilities"

print_step "Listing all available models"
print_demo "Show all configured models with their settings"
echo "Command: ${LF_CMD} models list"
${LF_CMD} models list

print_step "Showing details of the primary model"
echo "Command: ${LF_CMD} models show primary"
${LF_CMD} models show primary || true

print_step "Importing available Ollama models"
print_demo "Auto-discover and configure Ollama models"
echo "Command: ${LF_CMD} models import-ollama"
${LF_CMD} models import-ollama || true

print_success "Model management demonstrated"

# ================================================================
# Part 2: Multi-Database Setup
# ================================================================

print_header "Part 2: Setting Up Multiple Specialized Databases"

print_step "Backing up current configuration..."
cp "$PROJECT_CONFIG" "${PROJECT_CONFIG}.multi_demo_backup_$(date +%s)"

# Add multiple databases using UV and Python
uv run python << EOF
import yaml
import sys

config_file = "${PROJECT_CONFIG}"

# Read the current config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Define specialized databases
databases = [
    {
        'name': '${TECHNICAL_DB}',
        'type': 'ChromaStore',
        'config': {
            'collection_name': 'technical_documents',
            'distance_function': 'cosine',
            'persist_directory': './data/${TECHNICAL_DB}',
            'port': 8000
        },
        'embedding_strategies': [
            {
                'name': 'technical_embeddings',
                'type': 'OllamaEmbedder',
                'config': {
                    'auto_pull': True,
                    'base_url': 'http://localhost:11434/',
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
                'name': 'precise_search',
                'type': 'BasicSimilarityStrategy',
                'config': {
                    'distance_metric': 'cosine',
                    'top_k': 5
                },
                'default': True
            }
        ],
        'default_embedding_strategy': 'technical_embeddings',
        'default_retrieval_strategy': 'precise_search'
    },
    {
        'name': '${CREATIVE_DB}',
        'type': 'ChromaStore',
        'config': {
            'collection_name': 'creative_content',
            'distance_function': 'cosine',
            'persist_directory': './data/${CREATIVE_DB}',
            'port': 8001
        },
        'embedding_strategies': [
            {
                'name': 'creative_embeddings',
                'type': 'OllamaEmbedder',
                'config': {
                    'auto_pull': True,
                    'base_url': 'http://localhost:11434/',
                    'batch_size': 8,
                    'dimension': 768,
                    'model': 'nomic-embed-text',
                    'timeout': 60
                },
                'priority': 0
            }
        ],
        'retrieval_strategies': [
            {
                'name': 'broad_search',
                'type': 'BasicSimilarityStrategy',
                'config': {
                    'distance_metric': 'cosine',
                    'top_k': 15
                },
                'default': True
            }
        ],
        'default_embedding_strategy': 'creative_embeddings',
        'default_retrieval_strategy': 'broad_search'
    },
    {
        'name': '${RESEARCH_DB}',
        'type': 'ChromaStore',
        'config': {
            'collection_name': 'research_papers',
            'distance_function': 'cosine',
            'persist_directory': './data/${RESEARCH_DB}',
            'port': 8002
        },
        'embedding_strategies': [
            {
                'name': 'academic_embeddings',
                'type': 'OllamaEmbedder',
                'config': {
                    'auto_pull': True,
                    'base_url': 'http://localhost:11434/',
                    'batch_size': 12,
                    'dimension': 768,
                    'model': 'nomic-embed-text',
                    'timeout': 60
                },
                'priority': 0
            }
        ],
        'retrieval_strategies': [
            {
                'name': 'scholarly_search',
                'type': 'MetadataFilteredStrategy',
                'config': {
                    'fallback_multiplier': 2,
                    'filter_mode': 'post',
                    'top_k': 10
                },
                'default': True
            }
        ],
        'default_embedding_strategy': 'academic_embeddings',
        'default_retrieval_strategy': 'scholarly_search'
    }
]

# Add databases if they don't exist
for db in databases:
    db_exists = any(existing['name'] == db['name'] for existing in config.get('rag', {}).get('databases', []))
    if not db_exists:
        if 'rag' not in config:
            config['rag'] = {'databases': []}
        if 'databases' not in config['rag']:
            config['rag']['databases'] = []
        config['rag']['databases'].append(db)
        print(f"✓ Added database '{db['name']}'")
    else:
        print(f"ℹ Database '{db['name']}' already exists")

# Write back the updated config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("✓ All specialized databases configured")
EOF

print_success "Multiple databases configured"

# ================================================================
# Part 3: Create and Populate Datasets
# ================================================================

print_header "Part 3: Creating Specialized Datasets"

print_step "Creating technical documentation dataset"
echo "Command: ${LF_CMD} datasets add ${TECHNICAL_DATASET} -s universal_processor -b ${TECHNICAL_DB}"
${LF_CMD} datasets add "${TECHNICAL_DATASET}" -s universal_processor -b "${TECHNICAL_DB}"

print_step "Creating creative content dataset"
echo "Command: ${LF_CMD} datasets add ${CREATIVE_DATASET} -s universal_processor -b ${CREATIVE_DB}"
${LF_CMD} datasets add "${CREATIVE_DATASET}" -s universal_processor -b "${CREATIVE_DB}"

print_step "Creating research papers dataset"
echo "Command: ${LF_CMD} datasets add ${RESEARCH_DATASET} -s universal_processor -b ${RESEARCH_DB}"
${LF_CMD} datasets add "${RESEARCH_DATASET}" -s universal_processor -b "${RESEARCH_DB}"

# Ingest content into appropriate datasets
print_step "Ingesting technical documentation and code"
echo "Command: ${LF_CMD} datasets ingest ${TECHNICAL_DATASET} ${SAMPLE_DIR}/code_documentation/*.md ${SAMPLE_DIR}/code/*.py"
${LF_CMD} datasets ingest "${TECHNICAL_DATASET}" ${SAMPLE_DIR}/code_documentation/*.md ${SAMPLE_DIR}/code/*.py

print_step "Ingesting creative content (news articles)"
echo "Command: ${LF_CMD} datasets ingest ${CREATIVE_DATASET} ${SAMPLE_DIR}/news_articles/*.txt"
${LF_CMD} datasets ingest "${CREATIVE_DATASET}" ${SAMPLE_DIR}/news_articles/*.txt || true

print_step "Ingesting research papers"
echo "Command: ${LF_CMD} datasets ingest ${RESEARCH_DATASET} ${SAMPLE_DIR}/research_papers/*.txt"
${LF_CMD} datasets ingest "${RESEARCH_DATASET}" ${SAMPLE_DIR}/research_papers/*.txt

# Process all datasets
print_step "Processing all datasets into vector databases"
echo "Command: ${LF_CMD} datasets process ${TECHNICAL_DATASET}"
${LF_CMD} datasets process "${TECHNICAL_DATASET}"

echo "Command: ${LF_CMD} datasets process ${CREATIVE_DATASET}"
${LF_CMD} datasets process "${CREATIVE_DATASET}"

echo "Command: ${LF_CMD} datasets process ${RESEARCH_DATASET}"
${LF_CMD} datasets process "${RESEARCH_DATASET}"

print_success "All datasets created and processed"

# ================================================================
# Part 4: Demonstrate Model Selection for Different Tasks
# ================================================================

print_header "Part 4: Using Different Models for Different Tasks"

print_demo "Different models excel at different tasks. Let's demonstrate:"

print_step "Technical Query with Primary Model"
echo -e "${CYAN}Using the primary model for a technical programming question${NC}"
QUERY="How do I implement a REST API endpoint?"
echo "Command: timeout 15 ${LF_CMD} run --model primary --database ${TECHNICAL_DB} \"${QUERY}\""
echo -e "${GREEN}Response from PRIMARY model (optimized for accuracy):${NC}"
timeout 15 ${LF_CMD} run --model primary --database "${TECHNICAL_DB}" "${QUERY}" || true

echo -e "\n${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"

print_step "Creative Query with Creative Model (if available)"
echo -e "${CYAN}Using a creative model for story generation${NC}"
CREATIVE_QUERY="Write a short story opening about a programmer"
echo "Command: timeout 15 ${LF_CMD} run --model creative --no-rag \"${CREATIVE_QUERY}\""
echo -e "${MAGENTA}Response from CREATIVE model (high temperature):${NC}"
timeout 15 ${LF_CMD} run --model creative --no-rag "${CREATIVE_QUERY}" 2>/dev/null || {
    echo -e "${YELLOW}Creative model not configured, using primary${NC}"
    timeout 15 ${LF_CMD} run --no-rag "${CREATIVE_QUERY}" || true
}

print_success "Model selection demonstrated"

# ================================================================
# Part 5: Demonstrate Database Selection for Content Types
# ================================================================

print_header "Part 5: Using Different Databases for Different Content"

print_demo "Each database specializes in different content types"

print_step "Query Technical Database for Code Examples"
TECH_QUERY="Show me the DataProcessor class implementation"
echo -e "${CYAN}Searching technical database for code examples${NC}"
echo "Command: ${LF_CMD} rag query --database ${TECHNICAL_DB} \"${TECH_QUERY}\""
${LF_CMD} rag query --database "${TECHNICAL_DB}" --top-k 3 "${TECH_QUERY}"

echo -e "\n${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"

print_step "Query Research Database for Academic Content"
RESEARCH_QUERY="What are transformer attention mechanisms?"
echo -e "${CYAN}Searching research database for academic papers${NC}"
echo "Command: ${LF_CMD} rag query --database ${RESEARCH_DB} \"${RESEARCH_QUERY}\""
${LF_CMD} rag query --database "${RESEARCH_DB}" --top-k 5 "${RESEARCH_QUERY}"

print_success "Database routing demonstrated"

# ================================================================
# Part 6: Advanced Combination - Model + Database Selection
# ================================================================

print_header "Part 6: Combining Model and Database Selection"

print_demo "Optimal results by pairing the right model with the right database"

print_step "Technical Analysis: Primary Model + Technical Database"
ANALYSIS_QUERY="Explain the API design patterns in our codebase"
echo -e "${CYAN}Using PRIMARY model with TECHNICAL database for code analysis${NC}"
echo "Command: timeout 15 ${LF_CMD} run --model primary --database ${TECHNICAL_DB} \"${ANALYSIS_QUERY}\""
timeout 15 ${LF_CMD} run --model primary --database "${TECHNICAL_DB}" "${ANALYSIS_QUERY}" || true

echo -e "\n${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"

print_step "Research Summary: Primary Model + Research Database"
SUMMARY_QUERY="Summarize the key concepts in neural scaling laws"
echo -e "${CYAN}Using PRIMARY model with RESEARCH database for academic summary${NC}"
echo "Command: timeout 15 ${LF_CMD} run --model primary --database ${RESEARCH_DB} \"${SUMMARY_QUERY}\""
timeout 15 ${LF_CMD} run --model primary --database "${RESEARCH_DB}" "${SUMMARY_QUERY}" || true

print_success "Advanced combination demonstrated"

# ================================================================
# Part 7: Comparison - Same Query, Different Databases
# ================================================================

print_header "Part 7: Impact of Database Selection"

print_demo "Same query against different databases shows specialization"

COMPARISON_QUERY="What are the best practices?"

print_step "Querying Technical Database"
echo -e "${CYAN}Technical Database Response:${NC}"
echo "Command: ${LF_CMD} rag query --database ${TECHNICAL_DB} --top-k 2 \"${COMPARISON_QUERY}\""
${LF_CMD} rag query --database "${TECHNICAL_DB}" --top-k 2 "${COMPARISON_QUERY}"

echo -e "\n${CYAN}────────────────────────────────────────────────────────────────────────${NC}\n"

print_step "Querying Research Database"
echo -e "${CYAN}Research Database Response:${NC}"
echo "Command: ${LF_CMD} rag query --database ${RESEARCH_DB} --top-k 2 \"${COMPARISON_QUERY}\""
${LF_CMD} rag query --database "${RESEARCH_DB}" --top-k 2 "${COMPARISON_QUERY}"

print_success "Database specialization demonstrated"

# ================================================================
# Part 8: Performance Testing
# ================================================================

print_header "Part 8: Performance & Duplicate Detection"

print_step "Re-processing to show duplicate detection"
echo -e "${YELLOW}Re-processing datasets should skip all files as duplicates${NC}"
echo "Command: ${LF_CMD} datasets process ${TECHNICAL_DATASET}"
${LF_CMD} datasets process "${TECHNICAL_DATASET}"

print_success "Duplicate detection working correctly"

# ================================================================
# Summary
# ================================================================

print_header "✨ Demo Complete!"
echo -e "${GREEN}This demo showcased LlamaFarm's advanced capabilities:${NC}"
echo ""
echo "📊 Multi-Model Support:"
echo "  • Named model configurations (primary, creative, precise)"
echo "  • Easy model switching via --model flag"
echo "  • Import and auto-configure Ollama models"
echo ""
echo "🗄️ Multi-Database RAG:"
echo "  • Specialized databases for different content types"
echo "  • Technical DB for code and documentation"
echo "  • Research DB for academic papers"
echo "  • Creative DB for articles and stories"
echo ""
echo "🎯 Intelligent Routing:"
echo "  • Select models based on task requirements"
echo "  • Route to databases based on content type"
echo "  • Combine model + database for optimal results"
echo ""
echo "⚡ Production Features:"
echo "  • Efficient duplicate detection"
echo "  • Parallel processing capabilities"
echo "  • Flexible retrieval strategies"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}🚀 LlamaFarm: Multi-Model, Multi-Database AI Platform${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════════════${NC}"

# ================================================================
# Optional Cleanup
# ================================================================

echo ""
read -p "Do you want to clean up the demo databases and datasets? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Cleaning up demo resources..."
    
    # Remove datasets
    ${LF_CMD} datasets remove "${TECHNICAL_DATASET}" || true
    ${LF_CMD} datasets remove "${CREATIVE_DATASET}" || true
    ${LF_CMD} datasets remove "${RESEARCH_DATASET}" || true
    
    # Clean configuration using UV and Python
    uv run python << EOF
import yaml

config_file = "${PROJECT_CONFIG}"

# Read the current config
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Remove demo databases
demo_dbs = ['${TECHNICAL_DB}', '${CREATIVE_DB}', '${RESEARCH_DB}']
config['rag']['databases'] = [
    db for db in config.get('rag', {}).get('databases', [])
    if db['name'] not in demo_dbs
]

# Remove demo datasets
demo_datasets = ['${TECHNICAL_DATASET}', '${CREATIVE_DATASET}', '${RESEARCH_DATASET}']
config['datasets'] = [
    ds for ds in config.get('datasets', [])
    if ds['name'] not in demo_datasets
]

# Write back the updated config
with open(config_file, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("✓ Demo databases and datasets removed from configuration")
EOF
    
    print_success "Cleanup completed"
else
    print_info "Demo resources retained for further exploration"
    echo "To remove later, run:"
    echo "  ${LF_CMD} datasets remove ${TECHNICAL_DATASET}"
    echo "  ${LF_CMD} datasets remove ${CREATIVE_DATASET}"
    echo "  ${LF_CMD} datasets remove ${RESEARCH_DATASET}"
fi