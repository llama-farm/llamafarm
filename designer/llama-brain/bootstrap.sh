#!/bin/bash

# Bootstrap script for Llama Brain setup
# This script sets up the complete meta AI system

echo "🧠 Llama Brain Bootstrap"
echo "======================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run this from the llama-brain directory"
    exit 1
fi

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama not running. Please start it with: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Check required models
echo "🤖 Checking required models..."

# Check for llama3.2:3b
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "📥 Pulling llama3.2:3b..."
    ollama pull llama3.2:3b
fi

# Check for nomic-embed-text
if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 Pulling nomic-embed-text..."
    ollama pull nomic-embed-text
else
    echo "✅ Embedding model 'nomic-embed-text' available"
fi

echo "✅ Required models available"

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "📦 Installing dependencies with UV..."
    uv sync
fi

# Run the setup script
echo "🚀 Running setup script..."
uv run --active python setup.py

echo ""
echo "🎉 Bootstrap complete!"
echo ""
echo "Next steps:"
echo "  1. Start the server: uv run --active python -m llama_brain.server.main"
echo "  2. Start the test app: cd test-chat-app && python app.py"
echo "  3. Open browser to: http://localhost:5000"
echo ""
echo "Try asking: 'Create a RAG system for PDF documents'"