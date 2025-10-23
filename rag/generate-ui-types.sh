#!/bin/bash
set -e

echo "🦙 Generating RAG UI types from schema..."
echo ""

# Run the Python generator
uv run python generate-ui-types.py

echo ""
echo "✅ Done! Types generated in designer/src/components/Rag/generated/"
echo ""
echo "Next steps:"
echo "  1. Review the generated ragTypes.ts file"
echo "  2. Update UI components to import from generated types"
echo "  3. Commit the generated file to git"
