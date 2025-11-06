#!/usr/bin/env bash
#
# Generate TypeScript types from schemas
#
# This script runs type generators that create:
# - modelCatalog.ts (Model families and variants from models/schema.yaml)
# - ragTypes.ts (Parser and Extractor types from rag/schema.yaml)
# - databaseTypes.ts (Vector Store, Embedder, and Retrieval Strategy types from rag/schema.yaml)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "🦙 Generating Designer UI types from schemas..."
echo ""

# Generate model catalog types
npx tsx generate-model-types.ts

echo ""

# Generate RAG types
npx tsx generate-types.ts

echo ""
echo "✅ Done! Types generated in designer/src/types/"
echo "   - modelCatalog.ts (from models/schema.yaml)"
echo "   - ragTypes.ts (from rag/schema.yaml)"
echo "   - databaseTypes.ts (from rag/schema.yaml)"
