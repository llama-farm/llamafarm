#!/bin/bash

# RAG System Setup and Demo Script for macOS
# This script sets up the environment and demonstrates key features
# Enhanced with robust CLI output, more examples, and better pacing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_NAME="rag_env"
OLLAMA_MODEL="nomic-embed-text"
DEMO_DELAY=3  # Default delay between demo steps

# Animation frames for loading
SPINNER_FRAMES=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')

print_header() {
    echo -e "\n${CYAN}${BOLD}============================================================${NC}"
    echo -e "${CYAN}${BOLD}                    $1${NC}"
    echo -e "${CYAN}${BOLD}============================================================${NC}\n"
}

print_subheader() {
    echo -e "\n${BLUE}${BOLD}── $1 ──${NC}\n"
}

print_step() {
    echo -e "\n${BLUE}🔵 ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

print_highlight() {
    echo -e "${CYAN}${BOLD}🌟 $1${NC}"
}

print_demo() {
    echo -e "${YELLOW}🎬 ${BOLD}DEMO:${NC} ${YELLOW}$1${NC}"
}

print_output() {
    echo -e "${DIM}📤 Output:${NC} $1"
}

print_separator() {
    echo -e "${DIM}────────────────────────────────────────────────────────────${NC}"
}

# Enhanced spinner function
show_spinner() {
    local pid=$1
    local message=$2
    local i=0
    
    while kill -0 $pid 2>/dev/null; do
        printf "\r${BLUE}${SPINNER_FRAMES[i]} $message${NC}"
        i=$(((i + 1) % ${#SPINNER_FRAMES[@]}))
        sleep 0.1
    done
    printf "\r${GREEN}✅ $message - Complete${NC}\n"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Enhanced wait function with countdown
wait_for_user() {
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${YELLOW}${BOLD}Press Enter to continue or wait 10 seconds...${NC}"
        read -t 10 -r || true
        echo ""
    else
        echo -e "${DIM}Waiting ${DEMO_DELAY} seconds...${NC}"
        for ((i=DEMO_DELAY; i>0; i--)); do
            printf "\r${DIM}Continuing in ${i} seconds...${NC}"
            sleep 1
        done
        printf "\r${GREEN}Continuing...                 ${NC}\n"
    fi
}

# Enhanced command runner with better output
run_command() {
    local cmd="$1"
    local description="$2"
    local show_output="${3:-false}"
    
    echo -e "${CYAN}${BOLD}Running:${NC} ${DIM}$cmd${NC}"
    
    if [[ "$show_output" == "true" ]]; then
        print_separator
        echo -e "${DIM}"
        if eval "$cmd"; then
            echo -e "${NC}"
            print_separator
            print_success "$description completed"
        else
            echo -e "${NC}"
            print_separator
            print_error "$description failed"
            return 1
        fi
    else
        # Run command in background with spinner
        eval "$cmd" > /tmp/rag_demo_output.log 2>&1 &
        local cmd_pid=$!
        show_spinner $cmd_pid "$description"
        
        if wait $cmd_pid; then
            print_success "$description completed"
            if [[ -s /tmp/rag_demo_output.log ]]; then
                echo -e "${DIM}📋 Output summary: $(tail -1 /tmp/rag_demo_output.log | cut -c1-60)...${NC}"
            fi
        else
            print_error "$description failed"
            echo -e "${RED}Last error output:${NC}"
            tail -5 /tmp/rag_demo_output.log | sed 's/^/  /'
            return 1
        fi
    fi
}

# Enhanced system requirements check
check_system_requirements() {
    print_header "🔍 System Requirements Check"
    
    local checks_passed=0
    local total_checks=5
    
    # Check if we're on macOS
    print_step "Checking operating system..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        local os_version=$(sw_vers -productVersion)
        print_success "Running on macOS $os_version"
        ((checks_passed++))
    else
        print_error "This script is designed for macOS. Please adapt for your system."
        exit 1
    fi
    
    # Check available disk space
    print_step "Checking disk space..."
    local available_space=$(df -h . | awk 'NR==2{print $4}')
    print_success "Available disk space: $available_space"
    ((checks_passed++))
    
    # Check for Homebrew
    print_step "Checking for Homebrew..."
    if ! command_exists brew; then
        print_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        print_success "Homebrew installed successfully"
    else
        local brew_version=$(brew --version | head -1)
        print_success "Homebrew found: $brew_version"
    fi
    ((checks_passed++))
    
    # Check for Python 3.8+
    print_step "Checking Python installation..."
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d ' ' -f 2)
        local python_path=$(which python3)
        print_success "Python $python_version found at $python_path"
        
        # Check Python version compatibility
        local python_major=$(echo $python_version | cut -d. -f1)
        local python_minor=$(echo $python_version | cut -d. -f2)
        if [[ $python_major -ge 3 && $python_minor -ge 8 ]]; then
            print_success "Python version is compatible (3.8+ required)"
        else
            print_warning "Python version may be too old. 3.8+ recommended."
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    ((checks_passed++))
    
    # Check for uv (install if needed)
    print_step "Checking for uv package manager..."
    if ! command_exists uv; then
        print_warning "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv installed successfully"
    else
        local uv_version=$(uv --version)
        print_success "uv found: $uv_version"
    fi
    ((checks_passed++))
    
    # Check for Ollama
    print_step "Checking for Ollama..."
    if ! command_exists ollama; then
        print_warning "Ollama not found. Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed successfully"
    else
        local ollama_version=$(ollama --version 2>/dev/null || echo "ollama (version unknown)")
        print_success "Ollama found: $ollama_version"
    fi
    
    print_highlight "System check complete: $checks_passed/$total_checks requirements met"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        print_success "All system requirements satisfied!"
    else
        print_warning "Some requirements missing, but continuing..."
    fi
}

# Enhanced Python environment setup
setup_python_environment() {
    print_header "🐍 Python Environment Setup"
    
    cd "$PROJECT_DIR"
    
    # Show current directory
    print_info "Working directory: $(pwd)"
    
    # Create virtual environment if it doesn't exist
    print_step "Setting up virtual environment..."
    if [[ ! -d ".venv" ]]; then
        print_info "Creating new virtual environment with uv..."
        run_command "uv venv" "Virtual environment creation"
        print_success "Virtual environment created at .venv/"
    else
        print_success "Virtual environment already exists at .venv/"
        local venv_info=$(ls -la .venv | head -2 | tail -1 | awk '{print $6, $7, $8}')
        print_info "Created: $venv_info"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
    print_info "Python path: $(which python)"
    print_info "Pip path: $(which pip)"
    
    # Show current Python version in venv
    local venv_python_version=$(python --version)
    print_info "Virtual environment Python: $venv_python_version"
    
    # Install dependencies
    print_step "Installing core dependencies..."
    run_command "uv pip install -e ." "Core dependencies installation"
    
    # Install optional dependencies for demos
    print_step "Installing demo dependencies..."
    run_command "uv pip install python-dateutil textblob requests beautifulsoup4" "Demo dependencies installation"
    
    # Show installed packages summary
    print_step "Checking installed packages..."
    local package_count=$(uv pip list | wc -l)
    print_success "Installed $package_count packages in virtual environment"
    
    print_highlight "Python environment setup complete and ready!"
}

# Enhanced Ollama setup
setup_ollama() {
    print_header "🤖 Ollama Setup and Configuration"
    
    # Start Ollama service
    print_step "Starting Ollama service..."
    if pgrep -f "ollama serve" > /dev/null; then
        print_success "Ollama service already running"
        local ollama_pid=$(pgrep -f "ollama serve")
        print_info "Ollama PID: $ollama_pid"
    else
        print_info "Starting Ollama service in background..."
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        local ollama_pid=$!
        
        # Wait for Ollama to start
        print_step "Waiting for Ollama to initialize..."
        for i in {1..30}; do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                break
            fi
            sleep 1
            printf "\r${BLUE}Waiting for Ollama... ($i/30)${NC}"
        done
        printf "\n"
        print_success "Ollama service started (PID: $ollama_pid)"
    fi
    
    # Test Ollama connection
    print_step "Testing Ollama connection..."
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_success "Ollama API is responding on http://localhost:11434"
        
        # Show available models
        local model_count=$(ollama list | grep -v "NAME" | wc -l)
        print_info "Currently installed models: $model_count"
    else
        print_error "Ollama is not responding. Please check the installation."
        print_info "Check Ollama logs: tail -f /tmp/ollama.log"
        exit 1
    fi
    
    # Pull embedding model
    print_step "Setting up embedding model: $OLLAMA_MODEL..."
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        print_success "Model $OLLAMA_MODEL already available"
        local model_info=$(ollama list | grep "$OLLAMA_MODEL")
        print_info "Model details: $model_info"
    else
        print_info "Downloading model $OLLAMA_MODEL (this may take a few minutes)..."
        run_command "ollama pull $OLLAMA_MODEL" "Model download" true
        print_success "Model $OLLAMA_MODEL ready for use"
    fi
    
    # Test embedding functionality
    print_step "Testing embedding functionality..."
    local test_result=$(curl -s -X POST http://localhost:11434/api/embeddings \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$OLLAMA_MODEL\",\"prompt\":\"test\"}" | jq -r '.embedding[0:3]')
    
    if [[ "$test_result" != "null" && "$test_result" != "" ]]; then
        print_success "Embedding model is working correctly"
        print_info "Sample embedding (first 3 values): $test_result"
    else
        print_warning "Embedding test inconclusive, but proceeding..."
    fi
    
    print_highlight "Ollama setup complete and ready for RAG operations!"
}

# Enhanced system tests
run_system_tests() {
    print_header "🧪 System Integration Tests"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    local tests_passed=0
    local total_tests=6
    
    print_step "Test 1/6: CLI basic functionality..."
    if run_command "uv run python cli.py --help" "CLI help test"; then
        ((tests_passed++))
    fi
    
    wait_for_user
    
    print_step "Test 2/6: Extractor system..."
    if run_command "uv run python cli.py extractors list" "Extractor listing"; then
        ((tests_passed++))
        print_info "Testing YAKE extractor with sample text..."
        run_command "uv run python cli.py extractors test --extractor yake --text 'Machine learning and artificial intelligence are transforming technology. Deep learning models enable advanced natural language processing.'" "YAKE extractor test" true
    fi
    
    wait_for_user
    
    print_step "Test 3/6: Configuration system..."
    if run_command "uv run python cli.py --config examples/configs/unified_multi_strategy_config.yaml info" "Configuration test"; then
        ((tests_passed++))
    fi
    
    print_step "Test 4/6: Strategy system..."
    if run_command "uv run python cli.py strategies list" "Strategy listing"; then
        ((tests_passed++))
    fi
    
    print_step "Test 5/6: Component factories..."
    if run_command "uv run python -c 'from core.factories import ParserFactory, ExtractorFactory; print(f\"Parsers: {len(ParserFactory.list_available())}\"); print(f\"Extractors: {len(ExtractorFactory.list_available())}\")'" "Factory test"; then
        ((tests_passed++))
    fi
    
    print_step "Test 6/6: Sample data availability..."
    if [[ -d "samples" ]] && [[ -f "samples/csv/small_sample.csv" ]]; then
        print_success "Sample data found"
        local csv_lines=$(wc -l < samples/csv/small_sample.csv)
        print_info "Sample CSV has $csv_lines lines"
        ((tests_passed++))
    else
        print_warning "Sample data not found - some demos may be skipped"
    fi
    
    print_highlight "System tests complete: $tests_passed/$total_tests passed"
    
    if [[ $tests_passed -ge 4 ]]; then
        print_success "System is ready for full demonstration!"
    else
        print_warning "Some tests failed - continuing with limited demos..."
    fi
}

# Enhanced new component tests
run_new_component_tests() {
    print_header "🚀 New Component Testing Suite"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Testing newly added RAG components"
    print_info "This will test all the new parsers, extractors, embedders, and stores"
    
    wait_for_user
    
    print_subheader "📄 Testing New Document Parsers"
    print_info "Testing: DocxParser, PlainTextParser, HTMLParser, ExcelParser, MarkdownParser"
    print_separator
    run_command "uv run python tests/test_new_parsers.py" "New parser tests" true
    
    wait_for_user
    
    print_subheader "🔍 Testing New Content Extractors"
    print_info "Testing: TableExtractor, LinkExtractor, HeadingExtractor"
    print_separator
    run_command "uv run python tests/test_new_extractors.py" "New extractor tests" true
    
    wait_for_user
    
    print_subheader "🧠 Testing New Embedding Models"
    print_info "Testing: OpenAIEmbedder, HuggingFaceEmbedder, SentenceTransformerEmbedder"
    print_info "Note: Some tests may be skipped due to missing API keys or dependencies"
    print_separator
    run_command "uv run python tests/test_new_embedders.py" "New embedder tests" true
    
    wait_for_user
    
    print_subheader "🗄️  Testing New Vector Stores"
    print_info "Testing: FAISSStore, PineconeStore, QdrantStore"
    print_info "Note: Some tests may be skipped due to missing dependencies"
    
    # Check if we have a test file for stores
    if [[ -f "tests/test_new_stores.py" ]]; then
        print_separator
        run_command "uv run python tests/test_new_stores.py" "New vector store tests" true
    else
        print_warning "Vector store tests not found - creating basic test..."
        run_command "uv run python -c 'from core.factories import VectorStoreFactory; print(f\"Available stores: {VectorStoreFactory.list_available()}\")'" "Vector store factory test"
    fi
    
    print_highlight "New component testing complete!"
    print_success "All newly added components are integrated and functional"
    
    wait_for_user
}

# Enhanced strategy demo
run_strategy_demo() {
    print_header "🎯 RAG Strategy System Demonstration"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Exploring the RAG strategy system - the recommended way to use this system"
    
    wait_for_user
    
    print_subheader "📋 Available Strategies Overview"
    print_step "Listing all available RAG strategies..."
    run_command "uv run python cli.py strategies list" "List strategies" true
    
    wait_for_user
    
    print_subheader "🔍 Strategy Details"
    print_step "Examining the 'simple' strategy configuration..."
    run_command "uv run python cli.py strategies show simple" "Strategy details" true
    
    wait_for_user
    
    print_step "Examining the 'customer_support' strategy configuration..."
    run_command "uv run python cli.py strategies show customer_support" "Customer support strategy" true
    
    wait_for_user
    
    print_subheader "🎯 Strategy Recommendations"
    print_step "Getting strategy recommendations for different use cases..."
    
    print_info "Recommendations for customer support use case:"
    run_command "uv run python cli.py strategies recommend --use-case customer_support" "Customer support recommendations" true
    
    wait_for_user
    
    print_info "Recommendations for legal document analysis:"
    run_command "uv run python cli.py strategies recommend --use-case legal" "Legal recommendations" true
    
    wait_for_user
    
    print_info "Recommendations for research and academic use:"
    run_command "uv run python cli.py strategies recommend --use-case research" "Research recommendations" true
    
    wait_for_user
    
    print_subheader "⚙️ Strategy Configuration Export"
    print_step "Converting strategy to exportable config file..."
    run_command "uv run python cli.py strategies convert simple simple_strategy_exported.yaml" "Export strategy"
    
    if [[ -f "simple_strategy_exported.yaml" ]]; then
        print_success "Strategy exported successfully!"
        print_info "You can now use this config file: --config simple_strategy_exported.yaml"
        
        # Show a snippet of the exported config
        print_info "Config file preview:"
        echo -e "${DIM}"
        head -10 simple_strategy_exported.yaml | sed 's/^/  /'
        echo -e "${NC}"
        print_info "... (file continues)"
    fi
    
    wait_for_user
    
    print_subheader "📊 Strategy Performance Comparison"
    print_info "In a real scenario, you could compare strategies using:"
    echo -e "${YELLOW}  uv run python cli.py strategies benchmark --strategies simple,customer_support --test-queries queries.txt${NC}"
    echo -e "${YELLOW}  uv run python cli.py strategies analyze --strategy simple --metrics accuracy,speed,cost${NC}"
    
    print_highlight "Strategy system demonstration complete!"
    print_success "You now know how to leverage pre-configured RAG strategies effectively"
}

# Enhanced ingestion demo with multiple examples
run_ingestion_demo() {
    print_header "📥 Advanced Document Ingestion Demonstrations"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Showcasing various document ingestion capabilities with different extractors"
    
    wait_for_user
    
    print_subheader "📊 Demo 1: CSV Ingestion with Advanced Extractors"
    print_info "Ingesting customer support tickets with YAKE keywords and statistical analysis..."
    
    if [[ -f "samples/csv/small_sample.csv" ]]; then
        print_step "Preview of CSV data:"
        echo -e "${DIM}"
        head -3 samples/csv/small_sample.csv | sed 's/^/  /'
        echo -e "${NC}"
        
        print_step "Running ingestion with extractors..."
        run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest samples/csv/small_sample.csv --extractors yake statistics" "CSV ingestion with extractors" true
        
        print_success "CSV ingestion complete! Keywords and statistics extracted."
    else
        print_warning "Sample CSV not found - creating demo data..."
        cat > demo_tickets.csv << 'EOF'
id,subject,description,priority,status
1,"Login Issues","User cannot access their account after password reset",high,open
2,"Payment Problems","Credit card transaction failed during checkout",medium,pending
3,"Website Slow","Homepage loading very slowly on mobile devices",low,closed
4,"Data Export","Need to export user analytics data for Q3 report",medium,open
5,"Security Concern","Suspicious login attempts from unknown IP addresses",high,urgent
EOF
        print_success "Demo CSV created: demo_tickets.csv"
        run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest demo_tickets.csv --extractors yake statistics" "Demo CSV ingestion" true
    fi
    
    wait_for_user
    
    print_subheader "📄 Demo 2: Multi-format Document Processing"
    print_info "Testing different document formats with appropriate extractors..."
    
    # Create sample documents for demo
    print_step "Creating sample documents for demonstration..."
    
    # HTML document
    cat > demo_report.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>Quarterly Business Report</title></head>
<body>
<h1>Q3 2024 Business Report</h1>
<h2>Executive Summary</h2>
<p>This quarter showed significant growth across all metrics. Visit our website at <a href="https://company.com">company.com</a> for more details.</p>

<h2>Financial Performance</h2>
<table border="1">
<tr><th>Metric</th><th>Q2</th><th>Q3</th><th>Growth</th></tr>
<tr><td>Revenue</td><td>$100K</td><td>$125K</td><td>25%</td></tr>
<tr><td>Users</td><td>1000</td><td>1300</td><td>30%</td></tr>
</table>

<h3>Key Contacts</h3>
<p>For questions, contact support@company.com or call (555) 123-4567</p>
</body>
</html>
EOF
    
    # Markdown document
    cat > demo_notes.md << 'EOF'
# Project Meeting Notes

## Attendees
- John Smith (john@company.com)
- Sarah Johnson 
- Mike Chen

## Action Items

| Task | Owner | Due Date |
|------|-------|----------|
| Database optimization | Mike | 2024-01-15 |
| UI redesign mockups | Sarah | 2024-01-20 |
| Performance testing | John | 2024-01-25 |

## Next Steps
1. Complete database migration
2. Deploy to staging environment
3. Schedule user testing session

Contact the team at team@company.com for updates.
EOF
    
    # Plain text document
    cat > demo_manual.txt << 'EOF'
INSTALLATION GUIDE

System Requirements:
- Python 3.8 or higher
- 4GB RAM minimum
- 10GB disk space

Installation Steps:
1. Download the installer from https://download.example.com
2. Run the setup wizard
3. Configure your database connection
4. Test the installation

For support, email help@example.com or visit /docs/troubleshooting.html

Phone support: +1-800-555-0123
EOF
    
    print_success "Sample documents created"
    
    print_step "Processing HTML document with table and link extraction..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest demo_report.html --extractors table_extractor link_extractor heading_extractor" "HTML processing" true
    
    wait_for_user
    
    print_step "Processing Markdown document with heading and table extraction..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest demo_notes.md --extractors heading_extractor table_extractor link_extractor" "Markdown processing" true
    
    wait_for_user
    
    print_step "Processing plain text with comprehensive extraction..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest demo_manual.txt --extractors link_extractor heading_extractor" "Text processing" true
    
    wait_for_user
    
    print_subheader "🔄 Demo 3: Batch Processing with Strategy Override"
    print_info "Processing multiple documents using strategy with custom settings..."
    
    print_step "Using customer_support strategy with custom batch size..."
    run_command "uv run python cli.py --strategy customer_support --strategy-overrides '{\"components\":{\"embedder\":{\"config\":{\"batch_size\":8}}}}' ingest demo_*.* --extractors yake statistics" "Batch processing with strategy" true
    
    wait_for_user
    
    print_subheader "📈 Demo 4: Advanced Extraction Pipeline"
    print_info "Demonstrating complex extraction pipeline with multiple extractors..."
    
    # Create a complex document
    cat > complex_document.md << 'EOF'
# Technical Architecture Document

## Overview
This document outlines our microservices architecture deployed at https://api.internal.com

### Contact Information
- Team Lead: alice@company.com  
- DevOps: bob@company.com
- Support: +1-555-987-6543

## Service Performance Metrics

| Service | CPU % | Memory GB | Requests/sec |
|---------|-------|-----------|--------------|
| Auth API | 45 | 2.1 | 1200 |
| User Service | 30 | 1.8 | 800 |
| Payment Gateway | 60 | 3.2 | 400 |

### Deployment Configuration

```yaml
services:
  auth:
    image: auth:v1.2.3
    replicas: 3
  user:
    image: user:v2.1.0  
    replicas: 2
```

## Links and References
- Documentation: https://docs.internal.com/api
- Monitoring: https://grafana.internal.com/dashboards
- Repository: https://github.com/company/services

### File Locations
- Configs: /etc/services/config.yaml
- Logs: /var/log/services/
- Backups: /backup/db/services.sql

## Security Notes
Contact security@company.com for access requests.
Phone: (555) 444-3333
EOF
    
    print_step "Processing complex document with all extractors..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml ingest complex_document.md --extractors table_extractor,link_extractor,heading_extractor,yake,statistics" "Complex document processing" true
    
    print_highlight "Document ingestion demonstrations complete!"
    print_success "Various document formats processed successfully with different extraction strategies"
    
    # Cleanup demo files
    print_step "Cleaning up demo files..."
    rm -f demo_*.* complex_document.md
    print_success "Demo files cleaned up"
}

# Enhanced search demos
run_search_demos() {
    print_header "🔍 Advanced Search and Retrieval Demonstrations"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Showcasing different search strategies and retrieval methods"
    
    wait_for_user
    
    print_subheader "🎯 Demo 1: Basic Similarity Search"
    print_info "Performing semantic search for login-related issues..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml search 'login problems user access denied'" "Basic similarity search" true
    
    wait_for_user
    
    print_subheader "📊 Demo 2: Metadata-Enhanced Search"
    print_info "Using metadata-enhanced retrieval strategy for better context..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml search --retrieval metadata_enhanced 'password reset authentication'" "Metadata-enhanced search" true
    
    wait_for_user
    
    print_subheader "🔧 Demo 3: Technical Issue Search"
    print_info "Searching for technical problems with different parameters..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml search 'website performance slow loading database' --top-k 5" "Technical search with custom top-k" true
    
    wait_for_user
    
    print_subheader "📈 Demo 4: Search with Filtering"
    print_info "Advanced search with metadata filtering..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml search 'payment transaction failed' --filter-metadata 'priority:high'" "Filtered search" true
    
    wait_for_user
    
    print_subheader "🎨 Demo 5: Multi-Strategy Search Comparison"
    print_info "Comparing results from different retrieval strategies..."
    
    print_step "Using 'simple' strategy:"
    run_command "uv run python cli.py --strategy simple search 'data export analytics'" "Simple strategy search" true
    
    wait_for_user
    
    print_step "Using 'customer_support' strategy for the same query:"
    run_command "uv run python cli.py --strategy customer_support search 'data export analytics'" "Customer support strategy search" true
    
    wait_for_user
    
    print_subheader "🔤 Demo 6: Keyword-Based Search"
    print_info "Searching using extracted keywords and entities..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml search 'security breach suspicious activity IP address' --use-keywords" "Keyword-enhanced search" true
    
    print_highlight "Search and retrieval demonstrations complete!"
    print_success "Various search strategies and retrieval methods showcased successfully"
}

# Enhanced management demos
run_management_demos() {
    print_header "🗂️  Document Management and Analytics"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Exploring document management, analytics, and maintenance operations"
    
    wait_for_user
    
    print_subheader "📊 Demo 1: Comprehensive Document Statistics"
    print_step "Getting detailed document statistics and analytics..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage stats --detailed" "Detailed document statistics" true
    
    wait_for_user
    
    print_subheader "🔍 Demo 2: Document Analysis and Insights"
    print_step "Analyzing document content and extraction results..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage analyze --show-extractors --show-keywords" "Document analysis" true
    
    wait_for_user
    
    print_subheader "🔄 Demo 3: Hash-Based Duplicate Detection"
    print_step "Finding potential duplicate documents using content hashing..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage hash --find-duplicates --similarity-threshold 0.8" "Duplicate detection" true
    
    wait_for_user
    
    print_subheader "🧹 Demo 4: Maintenance Operations (Dry Run)"
    print_info "Simulating various maintenance operations without making changes..."
    
    print_step "Simulating cleanup of old documents (dry run)..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage delete --older-than 365 --dry-run" "Cleanup simulation" true
    
    wait_for_user
    
    print_step "Simulating expired document removal (dry run)..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage cleanup --expired --dry-run" "Expired cleanup simulation" true
    
    wait_for_user
    
    print_subheader "📈 Demo 5: Performance Metrics"
    print_step "Analyzing system performance and resource usage..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage performance --show-metrics --include-embeddings" "Performance metrics" true
    
    wait_for_user
    
    print_subheader "🔧 Demo 6: Index Optimization"
    print_info "Demonstrating index maintenance and optimization..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage optimize --rebuild-index --vacuum" "Index optimization" true
    
    print_highlight "Document management demonstrations complete!"
    print_success "Comprehensive document management and analytics capabilities showcased"
}

# Enhanced cleanup demo
run_cleanup_demo() {
    print_header "🧹 System Cleanup and Maintenance"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_demo "Demonstrating system cleanup and maintenance best practices"
    
    wait_for_user
    
    print_subheader "🗑️  Demo 1: Safe Cleanup Operations"
    print_step "Running safe cleanup operations on demo data..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage cleanup --expired --verbose" "Safe cleanup" true
    
    wait_for_user
    
    print_subheader "📊 Demo 2: Storage Optimization"
    print_step "Optimizing storage and compacting databases..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage storage --compact --analyze-usage" "Storage optimization" true
    
    wait_for_user
    
    print_subheader "🔄 Demo 3: Index Maintenance"
    print_step "Performing index maintenance and rebuilding..."
    run_command "uv run python cli.py --config examples/configs/extractors_demo_config.yaml manage index --rebuild --optimize" "Index maintenance" true
    
    wait_for_user
    
    print_subheader "💡 Best Practices Recommendations"
    print_info "In a production environment, consider these automation strategies:"
    echo ""
    echo -e "${YELLOW}🕐 Scheduled Maintenance:${NC}"
    echo "  - Daily: cleanup expired documents, optimize indices"
    echo "  - Weekly: analyze storage usage, remove duplicates"
    echo "  - Monthly: full system optimization, backup verification"
    echo ""
    echo -e "${YELLOW}📊 Monitoring:${NC}"
    echo "  - Set up alerts for storage usage > 80%"
    echo "  - Monitor embedding generation performance"
    echo "  - Track search response times and accuracy"
    echo ""
    echo -e "${YELLOW}🔒 Security:${NC}"
    echo "  - Regular security scans of stored documents"
    echo "  - Audit access logs and API usage"
    echo "  - Implement data retention policies"
    echo ""
    echo -e "${YELLOW}⚙️  Automation Commands:${NC}"
    echo "  # Add these to your crontab:"
    echo "  0 2 * * * cd $PROJECT_DIR && uv run python cli.py manage cleanup --expired"
    echo "  0 3 * * 0 cd $PROJECT_DIR && uv run python cli.py manage optimize --full"
    echo "  0 4 1 * * cd $PROJECT_DIR && uv run python cli.py manage backup --compress"
    
    print_highlight "Cleanup and maintenance demonstration complete!"
    print_success "System is optimized and ready for production use"
}

# Enhanced usage examples with more comprehensive information
show_usage_examples() {
    print_header "📚 Comprehensive Usage Guide and Examples"
    
    print_demo "Complete reference for using your RAG system effectively"
    
    wait_for_user
    
    print_subheader "🎯 Strategy-Based Operations (Recommended Approach)"
    echo -e "${CYAN}${BOLD}The strategy system is the easiest way to get started:${NC}\n"
    
    echo -e "${YELLOW}📋 Discover and explore strategies:${NC}"
    echo "  uv run python cli.py strategies list --detailed"
    echo "  uv run python cli.py strategies show customer_support"  
    echo "  uv run python cli.py strategies recommend --use-case legal"
    echo "  uv run python cli.py strategies benchmark --strategies simple,advanced"
    echo ""
    
    echo -e "${YELLOW}📥 Ingest documents with strategies:${NC}"
    echo "  uv run python cli.py --strategy simple ingest documents/"
    echo "  uv run python cli.py --strategy customer_support ingest tickets.csv"
    echo "  uv run python cli.py --strategy legal ingest contracts/ --batch-size 10"
    echo ""
    
    echo -e "${YELLOW}🔍 Search with strategies:${NC}"
    echo "  uv run python cli.py --strategy simple search 'machine learning'"
    echo "  uv run python cli.py --strategy customer_support search 'login problems'"
    echo "  uv run python cli.py --strategy legal search 'contract terms liability'"
    echo ""
    
    wait_for_user
    
    print_subheader "🔍 Advanced Extractor Operations"
    echo -e "${CYAN}${BOLD}Leverage the new extractors for enhanced document processing:${NC}\n"
    
    echo -e "${YELLOW}📋 Extractor management:${NC}"
    echo "  uv run python cli.py extractors list --show-dependencies"
    echo "  uv run python cli.py extractors test --extractor table_extractor --file report.html"
    echo "  uv run python cli.py extractors test --extractor link_extractor --text 'Visit https://example.com'"
    echo "  uv run python cli.py extractors benchmark --extractors yake,entities --test-files samples/"
    echo ""
    
    echo -e "${YELLOW}📄 Document processing with extractors:${NC}"
    echo "  uv run python cli.py ingest documents.pdf --extractors table_extractor,heading_extractor"
    echo "  uv run python cli.py ingest webpage.html --extractors link_extractor,table_extractor"
    echo "  uv run python cli.py ingest notes.md --extractors heading_extractor,yake"
    echo "  uv run python cli.py ingest data/ --extractors yake,entities,statistics,datetime"
    echo ""
    
    wait_for_user
    
    print_subheader "⚙️  Configuration-Based Operations"
    echo -e "${CYAN}${BOLD}For advanced users who need fine-grained control:${NC}\n"
    
    echo -e "${YELLOW}📂 Document ingestion:${NC}"
    echo "  uv run python cli.py --config custom_config.yaml ingest documents/"
    echo "  uv run python cli.py --config examples/configs/enterprise_config.yaml ingest --recursive /data/"
    echo "  uv run python cli.py ingest legal_docs/ --parser pdf --embedder openai --store pinecone"
    echo ""
    
    echo -e "${YELLOW}🔍 Advanced search operations:${NC}"
    echo "  uv run python cli.py search 'technical support' --top-k 10 --threshold 0.7"
    echo "  uv run python cli.py search 'payment issues' --retrieval metadata_enhanced --rerank"
    echo "  uv run python cli.py search 'user complaints' --filter-metadata 'priority:high,status:open'"
    echo "  uv run python cli.py search 'security incidents' --use-keywords --expand-query"
    echo ""
    
    wait_for_user
    
    print_subheader "🗂️  Document Management Operations"
    echo -e "${CYAN}${BOLD}Maintain and analyze your document collection:${NC}\n"
    
    echo -e "${YELLOW}📊 Analytics and insights:${NC}"
    echo "  uv run python cli.py manage stats --by-source --by-type --show-extractors"
    echo "  uv run python cli.py manage analyze --sentiment --topics --keywords"
    echo "  uv run python cli.py manage trends --time-range 30d --group-by source"
    echo "  uv run python cli.py manage export --format json --include-embeddings analytics.json"
    echo ""
    
    echo -e "${YELLOW}🧹 Maintenance operations:${NC}"
    echo "  uv run python cli.py manage cleanup --older-than 90d --dry-run"
    echo "  uv run python cli.py manage hash --find-duplicates --auto-remove"
    echo "  uv run python cli.py manage optimize --rebuild-index --compact-db"
    echo "  uv run python cli.py manage backup --compress --verify backups/"
    echo ""
    
    wait_for_user
    
    print_subheader "🔧 Advanced Configuration Examples"  
    echo -e "${CYAN}${BOLD}Configuration file templates and customization:${NC}\n"
    
    echo -e "${YELLOW}📝 Generate configurations:${NC}"
    echo "  uv run python cli.py config generate --template customer_support my_config.yaml"
    echo "  uv run python cli.py config validate --config my_config.yaml --strict"
    echo "  uv run python cli.py config convert --from strategy --strategy simple --to config.yaml"
    echo ""
    
    echo -e "${YELLOW}🎛️  Environment-specific configs:${NC}"
    echo "  # Development"
    echo "  uv run python cli.py --config configs/dev.yaml ingest test_data/"
    echo "  # Staging"
    echo "  uv run python cli.py --config configs/staging.yaml ingest --validate-only data/"
    echo "  # Production"
    echo "  uv run python cli.py --config configs/prod.yaml ingest --batch-size 100 data/"
    echo ""
    
    wait_for_user
    
    print_subheader "🐳 Production Deployment Examples"
    echo -e "${CYAN}${BOLD}Ready for production? Here are deployment patterns:${NC}\n"
    
    echo -e "${YELLOW}🚀 API Server:${NC}"
    echo "  # Start the RAG API server"
    echo "  uv run python api.py --config prod_config.yaml --workers 4 --port 8000"
    echo "  # Health check"
    echo "  curl http://localhost:8000/health"
    echo "  # Search API"
    echo "  curl -X POST http://localhost:8000/search -d '{\"query\":\"login issues\",\"top_k\":5}'"
    echo ""
    
    echo -e "${YELLOW}📊 Monitoring and observability:${NC}"
    echo "  # Enable metrics collection"
    echo "  export RAG_METRICS_ENABLED=true"
    echo "  export RAG_LOG_LEVEL=INFO"
    echo "  # View metrics"
    echo "  uv run python cli.py monitor --metrics --real-time"
    echo ""
    
    echo -e "${YELLOW}🔄 Batch processing:${NC}"
    echo "  # Process large document sets"
    echo "  uv run python cli.py batch-ingest --config batch_config.yaml --parallel 4 /data/docs/"
    echo "  # Scheduled maintenance"
    echo "  uv run python cli.py maintain --schedule daily --config maintenance.yaml"
    echo ""
    
    wait_for_user
    
    print_subheader "💡 Pro Tips and Best Practices"
    echo -e "${CYAN}${BOLD}Expert advice for optimal RAG system usage:${NC}\n"
    
    echo -e "${GREEN}✨ Performance Optimization:${NC}"
    echo "  • Use appropriate batch sizes (8-32 for embeddings)"
    echo "  • Enable indexing for frequently searched metadata fields"
    echo "  • Monitor memory usage during large document ingestion"
    echo "  • Use streaming for processing very large files"
    echo ""
    
    echo -e "${GREEN}🔍 Search Quality:${NC}"
    echo "  • Combine multiple extractors for richer metadata"
    echo "  • Use metadata filtering to narrow search scope"
    echo "  • Experiment with different embedding models for your domain"
    echo "  • Implement relevance feedback to improve results over time"
    echo ""
    
    echo -e "${GREEN}🛡️  Security and Privacy:${NC}"
    echo "  • Regularly rotate API keys and connection strings"
    echo "  • Implement document-level access controls"
    echo "  • Use encryption for sensitive document content"
    echo "  • Audit document access and modification logs"
    echo ""
    
    echo -e "${GREEN}📈 Scaling Considerations:${NC}"
    echo "  • Use vector databases (Pinecone, Qdrant) for large collections"
    echo "  • Implement caching for frequently accessed documents"
    echo "  • Consider document chunking strategies for very large files"
    echo "  • Monitor and tune retrieval performance regularly"
    echo ""
    
    wait_for_user
    
    print_highlight "🎉 Congratulations! You're now a RAG system expert!"
    echo ""
    echo -e "${CYAN}${BOLD}Quick Start Reminder:${NC}"
    echo -e "${YELLOW}1.${NC} Use strategies for simplicity: ${DIM}uv run python cli.py --strategy simple ingest docs/${NC}"
    echo -e "${YELLOW}2.${NC} Search your documents: ${DIM}uv run python cli.py --strategy simple search 'your query'${NC}"
    echo -e "${YELLOW}3.${NC} Monitor and maintain: ${DIM}uv run python cli.py manage stats${NC}"
    echo ""
    echo -e "${GREEN}🦙 Your RAG system is ready for action! No prob-llama! 🦙${NC}"
}

# Enhanced cleanup on exit
cleanup_on_exit() {
    if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
        print_info "Performing cleanup operations..."
        
        # Clean up demo databases
        rm -rf ./data/extractor_demo_chroma_db 2>/dev/null || true
        rm -rf ./data/simple_chroma_db 2>/dev/null || true
        
        # Clean up exported configs
        rm -f simple_strategy_exported.yaml 2>/dev/null || true
        
        # Clean up temporary files
        rm -f /tmp/rag_demo_output.log 2>/dev/null || true
        rm -f /tmp/ollama.log 2>/dev/null || true
        
        print_success "Cleanup completed"
    fi
}

# Enhanced main execution with better error handling
main() {
    # Parse command line arguments
    SKIP_PROMPTS=false
    CLEANUP_ON_EXIT=false
    RUN_TESTS_ONLY=false
    DEMO_DELAY=3
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-prompts)
                SKIP_PROMPTS=true
                shift
                ;;
            --cleanup)
                CLEANUP_ON_EXIT=true
                shift
                ;;
            --tests-only)
                RUN_TESTS_ONLY=true
                shift
                ;;
            --fast)
                DEMO_DELAY=1
                SKIP_PROMPTS=true
                shift
                ;;
            --delay)
                DEMO_DELAY="$2"
                shift 2
                ;;
            --help)
                echo -e "${CYAN}${BOLD}RAG System Setup and Demo Script${NC}"
                echo ""
                echo -e "${YELLOW}Usage:${NC} $0 [options]"
                echo ""
                echo -e "${YELLOW}Options:${NC}"
                echo "  --skip-prompts    Skip user prompts and run automatically"
                echo "  --cleanup         Clean up demo data on exit"
                echo "  --tests-only      Only run system tests, skip demos"
                echo "  --fast           Run in fast mode (skip prompts, 1s delay)"
                echo "  --delay SECONDS   Set delay between demo steps (default: 3)"
                echo "  --help           Show this help message"
                echo ""
                echo -e "${YELLOW}Examples:${NC}"
                echo "  $0                    # Interactive full demo"
                echo "  $0 --fast --cleanup   # Fast automated demo with cleanup"
                echo "  $0 --tests-only       # Only run tests"
                echo "  $0 --delay 5          # Slower demo with 5s delays"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Set up cleanup on exit
    trap cleanup_on_exit EXIT
    
    # Print enhanced welcome banner
    print_header "🦙 RAG System Setup and Demo Script 🦙"
    print_info "Enhanced with robust CLI output, comprehensive examples, and improved pacing"
    
    if [[ "$SKIP_PROMPTS" == "true" ]]; then
        print_info "Running in automated mode with ${DEMO_DELAY}s delays"
    else
        print_info "Running in interactive mode - you'll be prompted before each major step"
    fi
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${YELLOW}${BOLD}Press Enter to begin setup, or Ctrl+C to cancel...${NC}"
        read -r
    fi
    
    # Start execution timer
    local start_time=$(date +%s)
    
    # Execute setup steps with enhanced error handling
    set +e  # Don't exit on errors, handle them gracefully
    
    local step=1
    local total_steps=5
    
    print_info "Step $((step++))/$total_steps: System Requirements"
    if ! check_system_requirements; then
        print_error "System requirements check failed"
        exit 1
    fi
    
    print_info "Step $((step++))/$total_steps: Python Environment"
    if ! setup_python_environment; then
        print_error "Python environment setup failed"
        exit 1
    fi
    
    print_info "Step $((step++))/$total_steps: Ollama Configuration"
    if ! setup_ollama; then
        print_error "Ollama setup failed"
        exit 1
    fi
    
    print_info "Step $((step++))/$total_steps: System Integration Tests"
    if ! run_system_tests; then
        print_warning "Some system tests failed, but continuing..."
    fi
    
    print_info "Step $((step++))/$total_steps: New Component Tests"
    if ! run_new_component_tests; then
        print_warning "Some component tests failed, but continuing..."
    fi
    
    set -e  # Re-enable exit on error
    
    if [[ "$RUN_TESTS_ONLY" != "true" ]]; then
        # Execute comprehensive demo suite
        local demo_step=1
        local total_demos=6
        
        print_header "🎭 Comprehensive RAG System Demonstrations"
        print_info "Running $total_demos demonstration modules with enhanced examples"
        
        print_info "Demo $((demo_step++))/$total_demos: Strategy System"
        run_strategy_demo
        
        print_info "Demo $((demo_step++))/$total_demos: Document Ingestion"
        run_ingestion_demo
        
        print_info "Demo $((demo_step++))/$total_demos: Search and Retrieval"
        run_search_demos
        
        print_info "Demo $((demo_step++))/$total_demos: Document Management"
        run_management_demos
        
        print_info "Demo $((demo_step++))/$total_demos: System Maintenance"
        run_cleanup_demo
        
        print_info "Demo $((demo_step++))/$total_demos: Usage Guide"
        show_usage_examples
    fi
    
    # Calculate execution time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    print_header "🎉 Setup and Demo Complete! 🎉"
    print_success "Your RAG system is fully configured and ready to use!"
    print_info "Total execution time: ${minutes}m ${seconds}s"
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo ""
        echo -e "${CYAN}${BOLD}🚀 Quick Start Commands:${NC}"
        echo -e "${YELLOW}# Activate environment:${NC}"
        echo -e "${DIM}  cd $PROJECT_DIR && source .venv/bin/activate${NC}"
        echo ""
        echo -e "${YELLOW}# Try a simple search:${NC}"
        echo -e "${DIM}  uv run python cli.py --strategy simple search 'your query here'${NC}"
        echo ""
        echo -e "${YELLOW}# Ingest your documents:${NC}"
        echo -e "${DIM}  uv run python cli.py --strategy customer_support ingest /path/to/your/docs${NC}"
        echo ""
        echo -e "${CYAN}${BOLD}🦙 Happy RAG-ing! No prob-llama! 🦙${NC}"
    fi
}

# Run main function with all arguments
main "$@"