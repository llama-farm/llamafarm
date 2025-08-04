#!/bin/bash

# Models System Setup and Demo Script for macOS
# This script sets up the environment and demonstrates model management features

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
VENV_NAME="models_env"
PARENT_ENV_FILE="../.env"

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
    
    # Check for Python 3.10+
    if command_exists python3; then
        python_version=$(python3 --version | cut -d ' ' -f 2)
        major_version=$(echo $python_version | cut -d '.' -f 1)
        minor_version=$(echo $python_version | cut -d '.' -f 2)
        
        if [[ $major_version -ge 3 && $minor_version -ge 10 ]]; then
            print_success "Python $python_version found (>= 3.10 required)"
        else
            print_error "Python 3.10+ required, found $python_version"
            print_info "Install Python 3.10+ with: brew install python@3.10"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.10+"
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
    
    # Check for curl (needed for API testing)
    if ! command_exists curl; then
        print_error "curl not found. Please install curl."
        exit 1
    else
        print_success "curl found"
    fi
}

# Function to setup Python environment
setup_python_environment() {
    print_header "Setting Up Python Environment"
    
    cd "$PROJECT_DIR"
    
    # Check if pyproject.toml exists
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found in $PROJECT_DIR"
        print_info "Current directory: $(pwd)"
        print_info "Expected files: pyproject.toml, cli.py"
        exit 1
    fi
    
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
    run_command "uv sync" "Dependencies installation"
    
    print_success "Python environment setup complete"
}

# Function to check environment configuration
check_environment_config() {
    print_header "Checking Environment Configuration"
    
    cd "$PROJECT_DIR"
    
    # Check if parent .env exists
    if [[ -f "$PARENT_ENV_FILE" ]]; then
        print_success "Found .env file in parent directory"
        
        # Check for OpenAI API key
        if grep -q "OPENAI_API_KEY=" "$PARENT_ENV_FILE" && ! grep -q "OPENAI_API_KEY=$" "$PARENT_ENV_FILE"; then
            print_success "OpenAI API key configured"
        else
            print_warning "OpenAI API key not configured"
            print_info "Add your OpenAI API key to ../.env for full functionality"
        fi
        
        # Check for other API keys
        for provider in ANTHROPIC TOGETHER GROQ COHERE; do
            if grep -q "${provider}_API_KEY=" "$PARENT_ENV_FILE" && ! grep -q "${provider}_API_KEY=$" "$PARENT_ENV_FILE"; then
                print_success "$provider API key configured"
            else
                print_info "$provider API key not configured (optional)"
            fi
        done
        
        # Check for Hugging Face token
        if grep -q "HF_TOKEN=" "$PARENT_ENV_FILE" && ! grep -q "HF_TOKEN=$" "$PARENT_ENV_FILE"; then
            print_success "Hugging Face token configured"
        else
            print_info "HF_TOKEN not configured (optional for Hugging Face models)"
        fi
        
        # Check for Ollama configuration
        if grep -q "OLLAMA_HOST=" "$PARENT_ENV_FILE"; then
            ollama_host=$(grep "OLLAMA_HOST=" "$PARENT_ENV_FILE" | cut -d '=' -f 2)
            print_success "Ollama host configured: $ollama_host"
        else
            print_info "Ollama host not configured (using default: localhost)"
        fi
        
    else
        print_warning ".env file not found in parent directory"
        print_info "Create ../.env with API keys for full functionality"
        print_info "Example:"
        echo "  OPENAI_API_KEY=your_key_here"
        echo "  ANTHROPIC_API_KEY=your_key_here"
        echo "  OLLAMA_HOST=localhost"
    fi
}

# Function to test local model providers
test_local_model_providers() {
    print_header "Testing Local Model Providers"
    
    # Test Ollama if available
    if command_exists ollama; then
        print_step "Testing Ollama connection..."
        
        # Start Ollama if not running
        if ! pgrep -f "ollama serve" > /dev/null; then
            print_info "Starting Ollama service..."
            nohup ollama serve > /dev/null 2>&1 &
            sleep 3
        fi
        
        # Test connection
        if curl -s http://localhost:11434/api/tags > /dev/null; then
            print_success "Ollama is responding"
            
            # List available models
            print_step "Checking available Ollama models..."
            if ollama list | grep -q "NAME"; then
                models_count=$(ollama list | tail -n +2 | wc -l | tr -d '[:space:]')
                if [[ $models_count -gt 0 ]]; then
                    print_success "Found $models_count local models available:"
                    ollama list
                else
                    print_info "No models found. You can pull models with:"
                    echo "  ollama pull llama3.2:3b    # Small model (2GB)"
                    echo "  ollama pull llama3.1:8b    # Medium model (4.9GB)"
                    echo "  ollama pull mistral:7b     # Alternative model (4.1GB)"
                fi
            else
                print_info "No models found. You can pull models with: ollama pull llama3.2:3b"
            fi
            
            # Test Ollama via CLI
            print_step "Testing Ollama CLI commands..."
            run_command "uv run python cli.py list-local" "Ollama model listing"
            
        else
            print_warning "Ollama not responding (this is optional)"
        fi
    else
        print_info "Ollama not installed (optional for local models)"
        print_info "Install with: curl -fsSL https://ollama.ai/install.sh | sh"
    fi
    
    # Test vLLM availability
    print_step "Checking vLLM availability..."
    if python3 -c "import vllm" 2>/dev/null; then
        print_success "vLLM is available"
        run_command "uv run python cli.py list-vllm" "vLLM model listing"
    else
        print_info "vLLM not installed (optional for high-performance inference)"
        print_info "Install with: uv add --optional vllm"
    fi
    
    # Test TGI endpoints
    print_step "Checking Text Generation Inference endpoints..."
    run_command "uv run python cli.py list-tgi" "TGI endpoint listing"
}

# Function to run system tests
run_system_tests() {
    print_header "Running System Tests"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Running unit tests..."
    run_command "uv run python -m pytest tests/test_models.py -v --tb=short" "Unit tests"
    
    print_step "Testing CLI help system..."
    run_command "uv run python cli.py --help" "CLI help test"
    
    print_step "Testing YAML configuration loading..."
    run_command "uv run python cli.py --config config/default.yaml validate-config" "Default YAML config validation"
    
    print_step "Testing development configuration..."
    run_command "uv run python cli.py --config config/development.yaml list" "Development config test"
    
    print_step "Testing production configuration..."
    run_command "uv run python cli.py --config config/production.yaml validate-config" "Production config validation"
    
    print_step "Testing Ollama local configuration..."
    run_command "uv run python cli.py --config config/ollama_local.yaml validate-config" "Ollama local config validation"
    
    # Test end-to-end integration if environment allows
    if [[ -f "$PARENT_ENV_FILE" ]] && grep -q "OPENAI_API_KEY=" "$PARENT_ENV_FILE" && ! grep -q "OPENAI_API_KEY=$" "$PARENT_ENV_FILE"; then
        print_step "Running end-to-end integration tests..."
        run_command "uv run python -m pytest tests/test_e2e.py -v --tb=short -m 'not integration or integration'" "E2E tests (mocked)"
    else
        print_info "Skipping E2E tests (API keys not configured)"
    fi
    
    print_success "System tests completed"
}

# Function to run configuration demos
run_configuration_demos() {
    print_header "Model Configuration Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: Generate cloud model configurations"
    print_info "Creating basic cloud model configuration..."
    run_command "uv run python cli.py generate-config --type basic --output demo_basic_config.yaml" "Basic config generation"
    
    wait_for_user
    
    print_step "Demo 2: Generate multi-provider configuration"
    print_info "Creating a multi-provider configuration with fallback chains..."
    run_command "uv run python cli.py generate-config --type multi --output demo_multi_config.yaml" "Multi-provider config generation"
    
    wait_for_user
    
    print_step "Demo 3: Generate Ollama local model configuration"
    print_info "Creating configuration for local Ollama models..."
    run_command "uv run python cli.py generate-ollama-config --output demo_ollama_config.yaml" "Ollama config generation"
    
    wait_for_user
    
    print_step "Demo 4: Generate Hugging Face model configuration"
    print_info "Creating configuration for Hugging Face models..."
    run_command "uv run python cli.py generate-hf-config --output demo_hf_config.yaml --models 'gpt2,distilgpt2'" "HF config generation"
    
    wait_for_user
    
    print_step "Demo 5: Generate local inference engines configuration"
    print_info "Creating configuration for vLLM, TGI, and other local engines..."
    run_command "uv run python cli.py generate-engines-config --output demo_engines_config.yaml --include-unavailable" "Local engines config generation"
    
    wait_for_user
    
    print_step "Demo 6: Generate production configuration"
    print_info "Creating a production-ready configuration with monitoring..."
    run_command "uv run python cli.py generate-config --type production --output demo_production_config.yaml" "Production config generation"
    
    wait_for_user
    
    print_step "Demo 7: Validate all configurations"
    print_info "Validating all generated configurations..."
    for config in demo_basic_config.yaml demo_multi_config.yaml demo_ollama_config.yaml demo_hf_config.yaml demo_engines_config.yaml demo_production_config.yaml; do
        if [[ -f "$config" ]]; then
            run_command "uv run python cli.py --config $config validate-config" "Validating $config"
        fi
    done
    
    print_success "Configuration demos completed"
}

# Function to run provider management demos
run_provider_demos() {
    print_header "Provider Management Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: List cloud model providers"
    print_info "Listing all configured cloud model providers..."
    run_command "uv run python cli.py --config demo_multi_config.yaml list --detailed" "Cloud provider listing"
    
    wait_for_user
    
    print_step "Demo 2: List local Ollama models"
    print_info "Listing locally available Ollama models..."
    run_command "uv run python cli.py list-local" "Ollama model listing"
    
    wait_for_user
    
    print_step "Demo 3: List Hugging Face models"
    print_info "Searching popular Hugging Face models..."
    if [[ -f "$PARENT_ENV_FILE" ]] && grep -q "HF_TOKEN=" "$PARENT_ENV_FILE" && ! grep -q "HF_TOKEN=$" "$PARENT_ENV_FILE"; then
        run_command "uv run python cli.py list-hf --search 'gpt2' --limit 5" "HF model search"
    else
        print_info "HF_TOKEN not configured, showing vLLM compatible models instead..."
        run_command "uv run python cli.py list-vllm" "vLLM compatible models"
    fi
    
    wait_for_user
    
    print_step "Demo 4: List local inference engines"
    print_info "Checking local inference engine availability..."
    run_command "uv run python cli.py list-tgi" "TGI endpoints"
    
    wait_for_user
    
    print_step "Demo 5: Health check all providers"
    print_info "Checking health status of all providers..."
    run_command "uv run python cli.py --config demo_multi_config.yaml health-check" "Health check"
    
    wait_for_user
    
    # Test cloud providers if keys are available
    if [[ -f "$PARENT_ENV_FILE" ]] && grep -q "OPENAI_API_KEY=" "$PARENT_ENV_FILE" && ! grep -q "OPENAI_API_KEY=$" "$PARENT_ENV_FILE"; then
        print_step "Demo 6: Test OpenAI provider"
        print_info "Testing OpenAI provider with a simple query..."
        run_command "uv run python cli.py --config config/default.yaml test openai_gpt4o_mini --query 'Hello, LlamaFarm! Please respond with exactly: Test successful.'" "OpenAI provider test"
        
        wait_for_user
        
        print_step "Demo 7: Compare cloud providers"
        print_info "Comparing responses from multiple cloud providers..."
        run_command "uv run python cli.py --config config/default.yaml compare --providers openai_gpt4o_mini,anthropic_claude_3_haiku --query 'What is machine learning in one sentence?'" "Provider comparison"
    else
        print_info "Skipping OpenAI tests (API key not configured)"
    fi
    
    # Test local providers if available
    if command_exists ollama && curl -s http://localhost:11434/api/tags > /dev/null; then
        print_step "Demo 8: Test local Ollama model"
        print_info "Testing a local Ollama model..."
        # Get first available model
        first_model=$(ollama list | tail -n +2 | head -1 | cut -d' ' -f1)
        if [[ -n "$first_model" ]]; then
            run_command "uv run python cli.py test-local $first_model --query 'Hello from Ollama!'" "Ollama model test"
        else
            print_info "No Ollama models available for testing"
        fi
        
        wait_for_user
    fi
    
    # Test Hugging Face integration if available
    if [[ -f "$PARENT_ENV_FILE" ]] && grep -q "HF_TOKEN=" "$PARENT_ENV_FILE" && ! grep -q "HF_TOKEN=$" "$PARENT_ENV_FILE"; then
        print_step "Demo 9: Test Hugging Face integration"
        print_info "Testing Hugging Face login and model download..."
        run_command "uv run python cli.py hf-login" "HF login test"
        
        wait_for_user
    fi
    
    print_success "Provider demos completed"
}

# Function to run monitoring demos
run_monitoring_demos() {
    print_header "Monitoring and Analytics Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: Provider performance metrics"
    print_info "Analyzing provider performance characteristics..."
    echo -e "${CYAN}Provider Performance Analysis:${NC}"
    echo "📊 OpenAI GPT-4o-mini: ~200-800ms latency, $0.00015/1K tokens"
    echo "📊 OpenAI GPT-3.5-turbo: ~150-600ms latency, $0.002/1K tokens"
    echo "📊 Anthropic Claude: ~300-1000ms latency, varies by model"
    echo "📊 Ollama (local): ~500-2000ms latency, free but resource intensive"
    
    wait_for_user
    
    print_step "Demo 2: Cost optimization strategies"
    print_info "Demonstrating cost-aware provider selection..."
    echo -e "${CYAN}Cost Optimization Strategies:${NC}"
    echo "💰 Use GPT-4o-mini for simple tasks (10x cheaper than GPT-4)"
    echo "💰 Implement intelligent fallback chains"
    echo "💰 Cache frequent responses"
    echo "💰 Use local models for development/testing"
    echo "💰 Monitor usage patterns for optimization"
    
    wait_for_user
    
    print_step "Demo 3: Fallback chain simulation"
    print_info "Simulating provider failover scenarios..."
    echo -e "${CYAN}Fallback Chain Example:${NC}"
    echo "1️⃣ Primary: OpenAI GPT-4 (high quality, high cost)"
    echo "2️⃣ Fallback: OpenAI GPT-4o-mini (good quality, low cost)"
    echo "3️⃣ Emergency: Ollama local model (basic quality, free)"
    echo ""
    echo "🔄 System automatically switches on:"
    echo "  - API errors or timeouts"
    echo "  - Rate limit exceeded"
    echo "  - Cost budget exceeded"
    
    print_success "Monitoring demos completed"
}

# Function to run integration demos
run_integration_demos() {
    print_header "Integration with RAG System Demo"
    
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    
    print_step "Demo 1: RAG + Models workflow simulation"
    print_info "Demonstrating end-to-end RAG + Models integration..."
    echo -e "${CYAN}Typical Workflow:${NC}"
    echo "1️⃣ User Query: 'What is llama care?'"
    echo "2️⃣ RAG Retrieval: Find relevant documents"
    echo "3️⃣ Context Assembly: Combine query + retrieved context"
    echo "4️⃣ Model Selection: Choose optimal model based on:"
    echo "   - Query complexity"
    echo "   - Required response quality"
    echo "   - Cost constraints"
    echo "   - Latency requirements"
    echo "5️⃣ Response Generation: Send to selected model"
    echo "6️⃣ Quality Assessment: Evaluate response quality"
    
    wait_for_user
    
    print_step "Demo 2: Configuration integration"
    print_info "Showing how models integrate with existing RAG configs..."
    
    # Check if RAG system exists
    if [[ -f "../rag/cli.py" ]]; then
        echo -e "${CYAN}RAG + Models Integration:${NC}"
        echo "📁 RAG config: ../rag/config_examples/"
        echo "📁 Models config: ./config/"
        echo "🔗 Shared .env: ../.env"
        echo "🔄 Cross-system communication via:"
        echo "   - Shared configuration formats"
        echo "   - Environment variables"
        echo "   - API endpoints"
        echo "   - CLI command chaining"
    else
        print_info "RAG system not found in expected location"
    fi
    
    wait_for_user
    
    print_step "Demo 3: Prompt + Models integration preview"
    print_info "Preview of Prompts + Models integration..."
    
    # Check if prompts system exists
    if [[ -f "../prompts/cli.py" ]]; then
        echo -e "${CYAN}Prompts + Models Integration:${NC}"
        echo "📝 Prompt template selection"
        echo "🎯 Context-aware prompt formatting"
        echo "🤖 Model-specific prompt optimization"
        echo "📊 A/B testing of prompt-model combinations"
        echo "🔄 Feedback loop for continuous improvement"
        
        # Show example command structure
        echo -e "\n${YELLOW}Example Integration Commands:${NC}"
        echo "# 1. Select prompt template"
        echo "cd ../prompts && uv run python cli.py show technical_analysis"
        echo ""
        echo "# 2. Format prompt with context"
        echo "cd ../prompts && uv run python cli.py test technical_analysis --variables '{\"topic\": \"llama care\"}'"
        echo ""
        echo "# 3. Send to optimal model"
        echo "cd ../models && uv run python cli.py test openai_gpt4 --query \"[formatted prompt]\""
    else
        print_info "Prompts system not found in expected location"
    fi
    
    print_success "Integration demos completed"
}

# Function to show usage examples
show_usage_examples() {
    print_header "Usage Examples and Next Steps"
    
    echo -e "${CYAN}Here are some useful commands to explore:${NC}\n"
    
    echo -e "${YELLOW}🔧 Configuration Management:${NC}"
    echo "  # Cloud models configuration"
    echo "  uv run python cli.py generate-config --type basic --output my_config.yaml"
    echo "  uv run python cli.py generate-config --type production --output prod_config.yaml"
    echo "  uv run python cli.py validate-config --config my_config.yaml"
    echo "  uv run python cli.py --config my_config.yaml list --detailed"
    echo ""
    echo "  # Local models configuration"
    echo "  uv run python cli.py generate-ollama-config --output ollama_config.yaml"
    echo "  uv run python cli.py generate-hf-config --output hf_config.yaml"
    echo "  uv run python cli.py generate-engines-config --output engines_config.yaml"
    echo ""
    
    echo -e "${YELLOW}🏥 Provider Health & Testing:${NC}"
    echo "  # Cloud providers"
    echo "  uv run python cli.py health-check"
    echo "  uv run python cli.py test openai_gpt4 --query 'Hello, world!'"
    echo "  uv run python cli.py compare --providers openai_gpt4,anthropic_claude --query 'Explain AI'"
    echo ""
    echo "  # Local Ollama models"
    echo "  uv run python cli.py list-local"
    echo "  uv run python cli.py test-local llama3.2:3b --query 'Hello from Ollama!'"
    echo "  uv run python cli.py pull llama3.2:1b"
    echo ""
    echo "  # Hugging Face models"
    echo "  uv run python cli.py hf-login"
    echo "  uv run python cli.py list-hf --search 'gpt2' --limit 10"
    echo "  uv run python cli.py download-hf gpt2"
    echo "  uv run python cli.py test-hf gpt2 --query 'Hello' --max-tokens 20"
    echo ""
    echo "  # Local inference engines"
    echo "  uv run python cli.py list-vllm"
    echo "  uv run python cli.py list-tgi"
    echo "  uv run python cli.py test-vllm 'model_name' --query 'Hello'"
    echo "  uv run python cli.py test-tgi --endpoint 'http://localhost:8080' --query 'Hello'"
    echo ""
    
    echo -e "${YELLOW}📊 Performance & Monitoring:${NC}"
    echo "  # Monitor response times and costs"
    echo "  # Set up alerts for high error rates"
    echo "  # Analyze usage patterns for optimization"
    echo ""
    
    echo -e "${YELLOW}🔗 Integration Commands:${NC}"
    echo "  # Combine with RAG system:"
    echo "  cd ../rag && uv run python cli.py search 'llama care' | \\"
    echo "    cd ../models && uv run python cli.py test openai_gpt4"
    echo ""
    echo "  # Use with prompts system:"
    echo "  cd ../prompts && uv run python cli.py test qa_template --variables '{\"context\": \"...\"}' | \\"
    echo "    cd ../models && uv run python cli.py test openai_gpt4"
    echo ""
    
    echo -e "${YELLOW}⚙️  Advanced Configuration:${NC}"
    echo "  - Edit config files to customize provider settings"
    echo "  - Set up fallback chains for reliability"
    echo "  - Configure rate limiting and cost controls"
    echo "  - Implement custom monitoring and alerting"
    echo ""
    
    echo -e "${CYAN}🦙 Your model management system is ready! No prob-llama with the setup! 🦙${NC}"
}

# Function to run cleanup
cleanup_demo_files() {
    if [[ "$CLEANUP_ON_EXIT" == "true" ]]; then
        print_info "Cleaning up demo files..."
        rm -f demo_*.yaml
        rm -f test_*.yaml
        rm -f ollama_config.yaml hf_config.yaml engines_config.yaml 2>/dev/null || true
        print_success "Demo files cleaned up"
    fi
}

# Function to handle cleanup on exit
cleanup_on_exit() {
    cleanup_demo_files
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
                echo "Models System Setup and Demo Script"
                echo ""
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --skip-prompts    Skip user prompts and run automatically"
                echo "  --cleanup         Clean up demo files on exit"
                echo "  --tests-only      Only run system tests, skip demos"
                echo "  --help           Show this help message"
                echo ""
                echo "Prerequisites:"
                echo "  - macOS (Darwin)"
                echo "  - Python 3.10+"
                echo "  - uv package manager"
                echo "  - API keys in ../.env (optional but recommended)"
                echo ""
                echo "The script will:"
                echo "  1. Check system requirements"
                echo "  2. Set up Python environment with uv"
                echo "  3. Check API key configuration"
                echo "  4. Run comprehensive tests"
                echo "  5. Demonstrate all model management features"
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
    
    print_header "🦙 Models System Setup and Demo Script 🦙"
    print_info "This script will set up the Models environment and run demonstrations"
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${YELLOW}Press Enter to begin setup, or Ctrl+C to cancel...${NC}"
        read -r
    fi
    
    # Execute setup steps
    check_system_requirements
    setup_python_environment
    check_environment_config
    test_local_model_providers
    run_system_tests
    
    if [[ "$RUN_TESTS_ONLY" != "true" ]]; then
        # Execute demo steps
        run_configuration_demos
        run_provider_demos
        run_monitoring_demos
        run_integration_demos
        show_usage_examples
    fi
    
    print_header "🎉 Setup and Demo Complete! 🎉"
    print_success "Your Models system is ready to use!"
    
    if [[ "$SKIP_PROMPTS" != "true" ]]; then
        echo -e "\n${CYAN}To activate the environment in the future, run:${NC}"
        echo -e "${YELLOW}  cd $PROJECT_DIR${NC}"
        echo -e "${YELLOW}  source .venv/bin/activate${NC}"
        echo ""
        echo -e "${CYAN}Then you can use any of the CLI commands shown above.${NC}"
        echo ""
        echo -e "${PURPLE}Pro tips:${NC}"
        echo "  • Configure API keys in ../.env for full functionality"
        echo "  • Use --config flag to specify different configurations"
        echo "  • Run health-check regularly to monitor provider status"
        echo "  • Set up cost monitoring for production usage"
    fi
}

# Run main function
main "$@"