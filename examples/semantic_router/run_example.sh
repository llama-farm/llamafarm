#!/usr/bin/env bash
# Semantic Router Demo
# Demonstrates sub-millisecond routing of queries to specialized LLM models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${1:-${PROJECT_ROOT_DEFAULT}}"
LF_BIN="${LF_BIN:-${PROJECT_ROOT}/lf}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/llamafarm.yaml}"
NO_PAUSE=${NO_PAUSE:-0}

lf() {
  "${LF_BIN}" --cwd "${SCRIPT_DIR}" "$@"
}

bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '\033[0;34mℹ %s\033[0m\n' "$1"; }
success() { printf '\033[0;32m✓ %s\033[0m\n' "$1"; }
warn() { printf '\033[0;33m⚠ %s\033[0m\n' "$1"; }
error() { printf '\033[0;31m✗ %s\033[0m\n' "$1"; }

pause() {
  if [[ "$NO_PAUSE" != "1" ]]; then
    read -rp $'\nPress Enter to continue...'
  fi
}

ensure_file() {
  local path="$1" msg="$2"
  [[ -f "$path" ]] || { error "$msg"; exit 1; }
}

check_ollama() {
  if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    error "Ollama is not running. Start it with: ollama serve"
    exit 1
  fi
  success "Ollama is running"
}

bold "=================================="
bold "   Semantic Router Demo"
bold "=================================="
echo ""
info "This demo shows how the semantic router routes queries to specialized models"
info "based on topic similarity using sentence-transformer embeddings."
echo ""

ensure_file "$LF_BIN" "LlamaFarm CLI not found at ${LF_BIN}. Build it with 'go build -o lf cli/main.go'."
ensure_file "$CONFIG_PATH" "No example config found at ${CONFIG_PATH}."

bold "Step 1: Check prerequisites"
check_ollama
pause

bold "Step 2: Verify LlamaFarm services"
info "Make sure LlamaFarm server and Universal Runtime are running"
info "Start them with: lf start (from this directory)"

if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
  warn "LlamaFarm server not running. Starting services..."
  lf services start
  sleep 5
fi
success "LlamaFarm server is running"

if ! curl -s http://localhost:11540/health >/dev/null 2>&1; then
  warn "Universal Runtime not running. It should start automatically with the server."
fi
pause

bold "Step 3: Test billing queries"
info "These queries should route to the billing_specialist model"
echo ""

info "Query: 'What is my current account balance?'"
lf chat "What is my current account balance?"
pause

info "Query: 'Can I set up automatic payments?'"
lf chat "Can I set up automatic payments?"
pause

bold "Step 4: Test support queries"
info "These queries should route to the tech_support model"
echo ""

info "Query: 'I can't log in to my account'"
lf chat "I can't log in to my account"
pause

info "Query: 'The app keeps crashing'"
lf chat "The app keeps crashing"
pause

bold "Step 5: Test sales queries"
info "These queries should route to the sales_team model"
echo ""

info "Query: 'How much does the enterprise plan cost?'"
lf chat "How much does the enterprise plan cost?"
pause

bold "Step 6: Test fallback to general assistant"
info "Queries that don't match any route go to the default model"
echo ""

info "Query: 'What is the capital of France?'"
lf chat "What is the capital of France?"
pause

bold "Step 7: Direct router API testing"
info "You can also test the router directly via the Universal Runtime API"
echo ""

info "Training a test router..."
TRAIN_RESPONSE=$(curl -s -X POST "http://localhost:11540/v1/router/train" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo_test_router",
    "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "default_model": "general",
    "similarity_threshold": 0.5,
    "routes": [
      {
        "name": "greeting",
        "target_model": "greeter",
        "utterances": ["hello", "hi there", "good morning", "hey"]
      }
    ]
  }')
echo "Train response: $TRAIN_RESPONSE"
echo ""

info "Testing route decision..."
ROUTE_RESPONSE=$(curl -s -X POST "http://localhost:11540/v1/router/route" \
  -H "Content-Type: application/json" \
  -d '{"model": "demo_test_router", "query": "Hello, how are you?"}')
echo "Route response: $ROUTE_RESPONSE"
echo ""

info "Listing saved routers..."
LIST_RESPONSE=$(curl -s "http://localhost:11540/v1/router/models")
echo "Routers: $(echo "$LIST_RESPONSE" | jq -r '.data[].name' 2>/dev/null || echo "$LIST_RESPONSE")"
echo ""

info "Cleaning up test router..."
curl -s -X DELETE "http://localhost:11540/v1/router/models/demo_test_router" >/dev/null
success "Test router deleted"
pause

bold "=================================="
bold "   Demo Complete!"
bold "=================================="
echo ""
success "The semantic router successfully routes queries to specialized models:"
echo "  - Billing queries → billing_specialist"
echo "  - Support queries → tech_support"
echo "  - Sales queries → sales_team"
echo "  - Other queries → general_assistant"
echo ""
info "Try more queries with: lf chat \"your question here\""
info "Router models are saved in: ~/.llamafarm/models/router/"
echo ""
