#!/bin/bash
# Validates that manually-maintained types.go has all fields from the schema

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_FILE="$SCRIPT_DIR/config_types_generated.go"
MANUAL_FILE="$SCRIPT_DIR/../cli/cmd/config/types.go"

if [ ! -f "$GENERATED_FILE" ]; then
    echo "❌ Generated reference file not found. Run ./generate-go-types.sh first"
    exit 1
fi

echo "Validating Go types..."
echo ""

# Extract type definitions and field counts
echo "=== Type Comparison ==="
echo ""
echo "Generated types (reference from schema):"
grep "^type.*struct" "$GENERATED_FILE" | wc -l | xargs echo "  Structs:"

echo ""
echo "Manual types (in types.go):"
grep "^type.*struct" "$MANUAL_FILE" | wc -l | xargs echo "  Structs:"

echo ""
echo "=== Field Count by Major Types ==="
echo ""

# Function to count fields in a type
count_fields() {
    local file=$1
    local type=$2
    local pattern=$3

    # Extract the struct definition and count non-empty lines
    sed -n "/^type $pattern struct {/,/^}/p" "$file" 2>/dev/null | grep -c "^\s*[A-Z]" || echo "0"
}

# Compare key types
echo "LlamaFarmConfig fields:"
echo "  Generated: $(count_fields "$GENERATED_FILE" "SchemaDerefYaml" "SchemaDerefYaml")"
echo "  Manual:    $(count_fields "$MANUAL_FILE" "LlamaFarmConfig" "LlamaFarmConfig")"
echo ""

echo "Dataset fields:"
echo "  Generated: $(count_fields "$GENERATED_FILE" "Datasets" "SchemaDerefYamlDatasetsElem")"
echo "  Manual:    $(count_fields "$MANUAL_FILE" "Dataset" "Dataset")"
echo ""

echo "Model fields:"
echo "  Generated: $(count_fields "$GENERATED_FILE" "Models" "SchemaDerefYamlRuntimeModelsElem")"
echo "  Manual:    $(count_fields "$MANUAL_FILE" "Model" "Model")"
echo ""

echo "RuntimeConfig fields:"
echo "  Generated: $(count_fields "$GENERATED_FILE" "Runtime" "SchemaDerefYamlRuntime")"
echo "  Manual:    $(count_fields "$MANUAL_FILE" "RuntimeConfig" "RuntimeConfig")"
echo ""

echo "RAGConfig fields:"
echo "  Generated: $(count_fields "$GENERATED_FILE" "Rag" "SchemaDerefYamlRag")"
echo "  Manual:    $(count_fields "$MANUAL_FILE" "RAGConfig" "RAGConfig")"
echo ""

echo "✅ Validation complete"
echo ""
echo "To see detailed differences, compare the files manually:"
echo "  vimdiff $GENERATED_FILE $MANUAL_FILE"
