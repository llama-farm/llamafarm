#!/bin/bash

# Simple RAG Demo Runner
# Runs the enhanced demo that shows real embedding and retrieval

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${CYAN}${BOLD}============================================================${NC}"
    echo -e "${CYAN}${BOLD}                    $1${NC}"
    echo -e "${CYAN}${BOLD}============================================================${NC}\n"
}

print_step() {
    echo -e "${BLUE}🔵 ${BOLD}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header "🦙 Enhanced RAG Demo Runner 🦙"

echo -e "${YELLOW}This demo will show you the complete RAG pipeline:${NC}"
echo -e "• Real document parsing and processing"
echo -e "• Actual embedding generation with Ollama"
echo -e "• Vector storage in ChromaDB"
echo -e "• Semantic search with detailed results"
echo -e "• Complete transparency in the embedding process"

echo -e "\n${BOLD}Prerequisites:${NC}"
echo -e "✓ Ollama must be running"
echo -e "✓ nomic-embed-text model must be available"

echo -e "\n${DIM}Press Enter to start the demo, or Ctrl+C to cancel...${NC}"
read -r

print_step "Checking Ollama availability..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_error "Ollama is not running or not accessible"
    echo -e "Please start Ollama with: ${CYAN}ollama serve${NC}"
    exit 1
fi

# Check if the model is available
if ! curl -s http://localhost:11434/api/tags | grep -q "nomic-embed-text"; then
    print_error "nomic-embed-text model not found"
    echo -e "Please install it with: ${CYAN}ollama pull nomic-embed-text${NC}"
    exit 1
fi

print_success "Ollama is running with nomic-embed-text model"

print_step "Running enhanced RAG demo..."

# Run the enhanced demo
if uv run python enhanced_demo.py; then
    print_success "Demo completed successfully!"
    echo -e "\n${CYAN}${BOLD}🎉 You've seen the complete RAG pipeline in action!${NC}"
    echo -e "\n${BOLD}Next steps:${NC}"
    echo -e "• Use the CLI to query your documents: ${CYAN}uv run python cli.py search 'your query'${NC}"
    echo -e "• Explore different document types in the samples/ directory"
    echo -e "• Check out the configuration files in examples/configs/"
else
    print_error "Demo failed"
    echo -e "Please check the error messages above and ensure all dependencies are installed."
    exit 1
fi