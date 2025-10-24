#!/usr/bin/env bash
#
# Generate TypeScript types from rag/schema.yaml
#
# This script runs the unified type generator that creates:
# - ragTypes.ts (Parser and Extractor types)
# - databaseTypes.ts (Vector Store, Embedder, and Retrieval Strategy types)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

echo "🦙 Generating Designer UI types from schema..."
echo ""
npx tsx generate-types.ts

echo ""
echo "✅ Done! Types generated in designer/src/types/"
