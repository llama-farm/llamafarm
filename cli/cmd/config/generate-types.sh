#!/usr/bin/env sh
set -e

# Generate Go types using go-jsonschema
if ! command -v go-jsonschema >/dev/null 2>&1; then
    echo "Error: go-jsonschema not found. Install with: go install github.com/atombender/go-jsonschema@latest" >&2
    exit 1
fi

go-jsonschema -p config --struct-name-from-title -o types.go schema.json

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
