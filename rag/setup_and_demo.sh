#!/bin/bash

# RAG System Setup and Demo Script for macOS
# This script sets up the environment and demonstrates key features

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_NAME="rag_env"
OLLAMA_MODEL="nomic-embed-text"

print_header() {
    echo -e "\n${CYAN}============================================================${NC}"
    echo -e "${CYAN}                    $1${NC}"
    echo -e "${CYAN}============================================================${NC}\n"
}

print_step() {
    echo -e "\n${BLUE}🔵 $1${NC}"
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for user input
wait_for_user() {
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${YELLOW}Press Enter to continue...${NC}"
        read -r
    else
        sleep 2
    fi
}

# Function to run a command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${CYAN}Running: $cmd${NC}"
    if eval "$cmd"; then
        print_success "$description completed"
    else
        print_error "$description failed"
        return 1
    fi
}

# Function to check system requirements
check_system_requirements() {
    print_header "Checking System Requirements"
    
    # Check if we're on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_error "This script is designed for macOS. Please adapt for your system."
        exit 1
    fi
    print_success "Running on macOS"
    
    # Check for Homebrew
    if ! command_exists brew; then
        print_warning "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        print_success "Homebrew installed"
    else
        print_success "Homebrew found"
    fi
    
    # Check for Python 3.8+
    if command_exists python3; then
        python_version=$(python3 --version | cut -d ' ' -f 2)
        print_success "Python $python_version found"
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check for uv (install if needed)
    if ! command_exists uv; then
        print_warning "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
        print_success "uv installed"
    else
        print_success "uv found"
    fi
    
    # Check for Ollama
    if ! command_exists ollama; then
        print_warning "Ollama not found. Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed"
    else
        print_success "Ollama found"
    fi
}

# Function to setup Python environment
setup_python_environment() {
    print_header "Setting Up Python Environment"
    
    cd "$PROJECT_DIR"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        print_step "Creating virtual environment with uv..."
        run_command "uv venv" "Virtual environment creation"
    else
        print_success "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_step "Activating virtual environment..."
    source .venv/bin/activate
    print_success "Virtual environment activated"
    
    # Install dependencies
    print_step "Installing dependencies with uv..."
    run_command "uv pip install -e ." "Dependencies installation"
    
    # Install optional dependencies for better functionality
    print_step "Installing optional dependencies..."
    run_command "uv pip install python-dateutil textblob" "Optional dependencies installation"
    
    print_success "Python environment setup complete"
}

# Function to setup Ollama
setup_ollama() {
    print_header "Setting Up Ollama"
    
    # Start Ollama service
    print_step "Starting Ollama service..."
    if pgrep -f "ollama serve" > /dev/null; then
        print_success "Ollama service already running"
    else
        print_info "Starting Ollama in background..."
        nohup ollama serve > /dev/null 2>&1 &
        sleep 5
        print_success "Ollama service started"
    fi
    
    # Pull embedding model
    print_step "Pulling embedding model: $OLLAMA_MODEL..."
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        print_success "Model $OLLAMA_MODEL already available"
    else
        run_command "ollama pull $OLLAMA_MODEL" "Model download"
    fi
    
    # Test Ollama connection
    print_step "Testing Ollama connection..."
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        print_success "Ollama is responding"
    else
        print_error "Ollama is not responding. Please check the installation."
        exit 1
    fi
}

# Function to run system tests
run_system_tests() {
    print_header "Running System Tests"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Testing extractor system..."
    run_command "uv run python cli.py extractors list" "Extractor listing"
    
    print_step "Testing YAKE extractor..."
    run_command "uv run python cli.py extractors test --extractor yake --text 'Machine learning and artificial intelligence are transforming technology'" "YAKE extractor test"
    
    print_step "Testing configuration loading..."
    run_command "uv run python cli.py --config config_examples/unified_multi_strategy_config.json info" "Configuration test"
    
    print_success "System tests completed"
}

# Function to run ingestion demo
run_ingestion_demo() {
    print_header "RAG System Ingestion Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: CSV ingestion with extractors"
    print_info "Ingesting customer support tickets with YAKE and statistics extractors..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json ingest samples/csv/small_sample.csv --extractors yake statistics" "CSV ingestion with extractors"
    
    wait_for_user
    
    print_step "Demo 2: PDF ingestion"
    print_info "Ingesting PDF documents with entity extraction..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json ingest samples/pdfs/llama.pdf --extractors entities datetime" "PDF ingestion with extractors"
    
    wait_for_user
    
    print_step "Demo 3: Batch PDF ingestion with configuration-based extractors"
    print_info "Ingesting multiple PDFs using parser-configured extractors..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json ingest samples/pdfs/" "Batch PDF ingestion"
    
    print_success "Ingestion demos completed"
}

# Function to run search demos
run_search_demos() {
    print_header "RAG System Search Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: Basic similarity search"
    print_info "Searching for 'login problems'..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json search 'login problems'" "Basic search"
    
    wait_for_user
    
    print_step "Demo 2: Search with different strategy"
    print_info "Searching with metadata-enhanced strategy..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json search --retrieval metadata_enhanced 'password reset'" "Enhanced search"
    
    wait_for_user
    
    print_step "Demo 3: Technical search"
    print_info "Searching for technical issues..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json search 'website malfunction server error'" "Technical search"
    
    print_success "Search demos completed"
}

# Function to run document management demos
run_management_demos() {
    print_header "Document Management Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: Document statistics"
    print_info "Getting document statistics..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json manage stats --detailed" "Document statistics"
    
    wait_for_user
    
    print_step "Demo 2: Hash-based operations"
    print_info "Finding duplicate documents..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json manage hash --find-duplicates" "Duplicate detection"
    
    wait_for_user
    
    print_step "Demo 3: Soft deletion (dry run)"
    print_info "Simulating deletion of old documents..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json manage delete --older-than 365 --dry-run" "Soft deletion dry run"
    
    print_success "Management demos completed"
}

# Function to run cleanup
run_cleanup_demo() {
    print_header "Cleanup and Reset Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo: Cleanup operations"
    print_info "Running cleanup to remove old documents..."
    run_command "uv run python cli.py --config config_examples/extractors_demo_config.json manage cleanup --expired" "Cleanup expired documents"
    
    print_info "Note: In a real scenario, you might want to:"
    echo "  - Set up automated cleanup schedules"
    echo "  - Configure retention policies"
    echo "  - Implement backup strategies"
    echo "  - Monitor system performance"
    
    print_success "Cleanup demo completed"
}

# Function to show usage examples
show_usage_examples() {
    print_header "Usage Examples and Next Steps"
    
    echo -e "${CYAN}Here are some useful commands to explore:${NC}\n"
    
    echo -e "${YELLOW}📋 List and test extractors:${NC}"
    echo "  uv run python cli.py extractors list --detailed"
    echo "  uv run python cli.py extractors test --extractor entities --file samples/pdfs/llama.pdf"
    echo ""
    
    echo -e "${YELLOW}📂 Ingest documents:${NC}"
    echo "  uv run python cli.py ingest samples/csv/large_sample.csv --extractors yake entities statistics"
    echo "  uv run python cli.py --config config_examples/enterprise_document_management_config.json ingest samples/pdfs/"
    echo ""
    
    echo -e "${YELLOW}🔍 Search documents:${NC}"
    echo "  uv run python cli.py search 'technical support troubleshooting'"
    echo "  uv run python cli.py search --retrieval metadata_enhanced 'billing issues'"
    echo ""
    
    echo -e "${YELLOW}🗂️  Manage documents:${NC}"
    echo "  uv run python cli.py manage stats"
    echo "  uv run python cli.py manage delete --older-than 30 --strategy soft"
    echo "  uv run python cli.py manage replace document.pdf --target-doc-id doc123"
    echo ""
    
    echo -e "${YELLOW}⚙️  Advanced configuration:${NC}"
    echo "  - Edit config_examples/*.json to customize behavior"
    echo "  - Add custom extractors in extractors/ directory"
    echo "  - Create custom retrieval strategies"
    echo "  - Set up automated maintenance schedules"
    echo ""
    
    echo -e "${CYAN}🦙 The RAG system is ready for action! No prob-llama! 🦙${NC}"
}

# Function to handle cleanup on exit
cleanup_on_exit() {
    if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
        print_info "Cleaning up demo data..."
        rm -rf ./data/extractor_demo_chroma_db
        print_success "Demo data cleaned up"
    fi
}

# Main execution
main() {
    # Parse command line arguments
    SKIP_PROMPTS=false
    CLEANUP_ON_EXIT=false
    RUN_TESTS_ONLY=false
    
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
            --help)
                echo "RAG System Setup and Demo Script"
                echo ""
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-prompts    Skip user prompts and run automatically"
                echo "  --cleanup         Clean up demo data on exit"
                echo "  --tests-only      Only run system tests, skip demos"
                echo "  --help           Show this help message"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Set up cleanup on exit
    trap cleanup_on_exit EXIT
    
    print_header "🦙 RAG System Setup and Demo Script 🦙"
    print_info "This script will set up the RAG environment and run demonstrations"
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${YELLOW}Press Enter to begin setup, or Ctrl+C to cancel...${NC}"
        read -r
    fi
    
    # Execute setup steps
    check_system_requirements
    setup_python_environment
    setup_ollama
    run_system_tests
    
    if [[ "$RUN_TESTS_ONLY" != "true" ]]; then
        # Execute demo steps
        run_ingestion_demo
        run_search_demos
        run_management_demos
        run_cleanup_demo
        show_usage_examples
    fi
    
    print_header "🎉 Setup and Demo Complete! 🎉"
    print_success "Your RAG system is ready to use!"
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${CYAN}To activate the environment in the future, run:${NC}"
        echo -e "${YELLOW}  cd $PROJECT_DIR${NC}"
        echo -e "${YELLOW}  source .venv/bin/activate${NC}"
        echo ""
        echo -e "${CYAN}Then you can use any of the CLI commands shown above.${NC}"
    fi
}

# Run main function
main "$@"