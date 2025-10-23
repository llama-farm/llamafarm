#!/bin/bash
set -e

echo "🦙 Generating Database/Embedding UI types from schema..."
echo ""

# Run the Python generator
uv run python generate-db-embedding-types.py

echo ""
echo "✅ Done! Types generated in designer/src/components/Rag/generated/"
echo ""
echo "Generated types for:"
echo "  • Vector Stores (ChromaStore, QdrantStore, FAISSStore, PineconeStore)"
echo "  • Embedders (Ollama, HuggingFace, OpenAI, SentenceTransformer)"
echo "  • Retrieval Strategies (Basic, Filtered, MultiQuery, Reranked, Hybrid)"
echo ""
echo "Next steps:"
echo "  1. Review the generated databaseTypes.ts file"
echo "  2. Update Databases UI components to import from generated types"
echo "  3. Commit the generated file to git"
