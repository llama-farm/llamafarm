#!/bin/bash
set -e

# Generate Go types from JSON Schema using go-jsonschema for VALIDATION
# This generates a reference file to compare against our manually-crafted types.go
# The generated file helps ensure our manual types match the schema structure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED_FILE="$SCRIPT_DIR/config_types_generated.go"
MANUAL_FILE="$SCRIPT_DIR/../cli/cmd/config/types.go"
SCHEMA_FILE="$SCRIPT_DIR/schema.deref.yaml"

echo "Installing/updating go-jsonschema..."
go install github.com/atombender/go-jsonschema@v0.21.0

# Get GOPATH to find installed binary
GOPATH=$(go env GOPATH)
GOJSONSCHEMA="$GOPATH/bin/go-jsonschema"

echo "Generating Go types from schema for validation..."
"$GOJSONSCHEMA" \
  --package=config \
  --only-models \
  "$SCHEMA_FILE" > "$GENERATED_FILE"

echo ""
echo "✅ Generated reference types at $GENERATED_FILE"
echo ""
echo "Use this generated file to validate that $MANUAL_FILE"
echo "contains all required fields from the schema."
echo ""
echo "Note: The generated file has verbose type names (e.g., SchemaDerefYamlDatasetsElem)"
echo "but the field names and types should match what's in the manual types.go file."
echo ""
echo "To compare structures:"
echo "  grep 'type.*struct' $GENERATED_FILE"
echo "  grep 'type.*struct' $MANUAL_FILE"
