#!/usr/bin/env sh
set -e

# Ensure Go bin directories are in PATH (for CI environments)
export PATH="$HOME/go/bin:$GOPATH/bin:$PATH"

echo "DEBUG: Current directory: $(pwd)"
echo "DEBUG: PATH=$PATH"
echo "DEBUG: Looking for schema.json..."
ls -la schema.json 2>/dev/null || echo "DEBUG: schema.json NOT FOUND"

# Generate Go types using go-jsonschema
if ! command -v go-jsonschema >/dev/null 2>&1; then
    echo "Error: go-jsonschema not found. Install with: go install github.com/atombender/go-jsonschema@v0.21.0" >&2
    echo "DEBUG: Checking $HOME/go/bin..."
    ls -la "$HOME/go/bin" 2>/dev/null || echo "DEBUG: $HOME/go/bin not found"
    exit 1
fi

echo "DEBUG: go-jsonschema found at: $(which go-jsonschema)"
go-jsonschema -p config --struct-name-from-title -o types.go schema.json
echo "DEBUG: types.go generated, checking..."
ls -la types.go
grep -c "LlamaFarmConfigPromptsElem" types.go || echo "DEBUG: LlamaFarmConfigPromptsElem NOT FOUND in types.go"

# Verify --struct-name-from-title worked (PromptSet should exist, not LlamaFarmConfigPromptsElem)
if ! grep -q 'type PromptSet struct' types.go; then
    echo "Error: --struct-name-from-title flag did not work correctly." >&2
    echo "Expected 'type PromptSet struct' but it was not found in types.go" >&2
    echo "This usually means go-jsonschema version doesn't support this flag." >&2
    echo "Ensure go-jsonschema v0.21.0+ is installed." >&2
    exit 1
fi

# Fix go-jsonschema bug: when additionalProperties:true is combined with
# minimum constraints, it generates code that uses 'raw' variable without
# declaring it. This Python script adds the missing declaration only where needed.
# See: https://github.com/atombender/go-jsonschema/issues/XXX
if grep -q 'delete(raw, st.Field' types.go; then
    python3 - << 'PYEOF'
with open('types.go', 'r') as f:
    lines = f.readlines()

fixed = False
i = 0
while i < len(lines):
    # Look for the pattern: unmarshal into plain, then delete(raw,...) without raw declaration
    if 'if err := json.Unmarshal(value, &plain)' in lines[i]:
        # Check if this block uses raw without declaring it
        # Look ahead for delete(raw, within the next 20 lines
        has_delete_raw = False
        has_raw_decl = False
        for j in range(max(0, i-5), min(len(lines), i+20)):
            if 'var raw map[string]interface{}' in lines[j]:
                has_raw_decl = True
            if 'delete(raw,' in lines[j]:
                has_delete_raw = True

        if has_delete_raw and not has_raw_decl:
            # Insert raw declaration before the unmarshal line
            indent = '\t'
            new_lines = [
                indent + 'var raw map[string]interface{}\n',
                indent + 'if err := json.Unmarshal(value, &raw); err != nil {\n',
                indent + '\treturn err\n',
                indent + '}\n',
            ]
            lines = lines[:i] + new_lines + lines[i:]
            i += len(new_lines)
            fixed = True
    i += 1

with open('types.go', 'w') as f:
    f.writelines(lines)

if fixed:
    print("Fixed additionalProperties unmarshal bug")
else:
    print("No fix needed")
PYEOF
fi

# Clean up temporary JSON file
rm -f schema.json schema.yaml

echo "✓ Generated types.go from schema.yaml"
