#!/usr/bin/env sh
set -e

# Generate Go types using go-jsonschema
if ! command -v go-jsonschema >/dev/null 2>&1; then
    echo "Error: go-jsonschema not found. Install with: go install github.com/atombender/go-jsonschema@latest" >&2
    exit 1
fi

go-jsonschema -p config --struct-name-from-title -o types.go schema.json

# Clean up temporary JSON file
rm -f schema.json schema.yaml

echo "✓ Generated types.go from schema.yaml"
