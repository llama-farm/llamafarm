#!/bin/bash
# LlamaFarm Integration Demo Runner
# 
# This script demonstrates the complete integration of Models, RAG, and Prompts
# using the Llama Brain configurations as examples.

set -e  # Exit on any error

echo "🚀 LlamaFarm Integration Demo"
echo "=============================="

# Check if query is provided
if [ $# -eq 0 ]; then
    echo "❌ Please provide a query as an argument"
    echo ""
    echo "Usage: $0 \"Your question here\""
    echo ""
    echo "Example queries:"
    echo "  $0 \"How do I configure a local model?\""
    echo "  $0 \"What are RAG best practices?\""
    echo "  $0 \"How do I create custom prompts?\""
    echo "  $0 \"Show me a production setup example\""
    exit 1
fi

QUERY="$1"

echo "Query: $QUERY"
echo ""

# Check if we're in the right directory
if [ ! -f "demo_integration.py" ]; then
    echo "❌ Please run this script from the llama-brain directory"
    exit 1
fi

# Run the integration demo
echo "🔧 Running integration pipeline..."
echo ""

uv run --active python demo_integration.py "$QUERY"

echo ""
echo "✨ Demo completed!"
echo ""
echo "💡 What happened:"
echo "  1. RAG searched for relevant documents"
echo "  2. Prompts applied the configuration template"  
echo "  3. Models generated the final response"
echo ""
echo "🔧 To customize:"
echo "  • Edit configs in: ./configs/"
echo "  • Modify demo logic: ./demo_integration.py"
echo "  • Add your own documents to RAG"