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

# Fix go-jsonschema duplicate types: when the schema has cross-file $ref paths
# that resolve to the same definition (e.g., componentsDefinition referenced from
# both config/schema.yaml and rag/schema.yaml), go-jsonschema emits duplicate
# struct definitions and methods. This script removes the second occurrence.
python3 - << 'PYEOF'
import re

with open('types.go', 'r') as f:
    content = f.read()

# Find all type declarations: "type FooBar struct {"
type_pattern = re.compile(r'^type (\w+) struct \{', re.MULTILINE)
all_types = type_pattern.findall(content)

# Find duplicates
seen = set()
duplicates = set()
for t in all_types:
    if t in seen:
        duplicates.add(t)
    seen.add(t)

if not duplicates:
    print("No duplicate types found")
else:
    lines = content.split('\n')
    output_lines = []
    skip_until_closing = False
    skip_type_name = None
    first_seen = set()
    removed = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for duplicate type declaration
        type_match = re.match(r'^type (\w+) struct \{', line)
        if type_match:
            name = type_match.group(1)
            if name in duplicates:
                if name in first_seen:
                    # Skip this duplicate type block
                    skip_until_closing = True
                    skip_type_name = name
                    brace_depth = 1
                    i += 1
                    while i < len(lines) and brace_depth > 0:
                        brace_depth += lines[i].count('{') - lines[i].count('}')
                        i += 1
                    removed += 1
                    continue
                else:
                    first_seen.add(name)

        # Check for duplicate method declaration
        method_match = re.match(r'^func \(j \*(\w+)\) (\w+)\(', line)
        if method_match:
            name = method_match.group(1)
            sig = f"{name}.{method_match.group(2)}"
            if name in duplicates and name in first_seen:
                # Check if we already have this method
                method_key = f"func (j *{name}) {method_match.group(2)}"
                # Count how many times this method signature appears before this line
                preceding = '\n'.join(output_lines)
                if method_key in preceding:
                    # Skip duplicate method
                    brace_depth = 0
                    while i < len(lines):
                        brace_depth += lines[i].count('{') - lines[i].count('}')
                        if brace_depth <= 0 and '}' in lines[i]:
                            i += 1
                            break
                        i += 1
                    removed += 1
                    continue

        output_lines.append(line)
        i += 1

    if removed > 0:
        with open('types.go', 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Removed {removed} duplicate type/method declarations: {', '.join(sorted(duplicates))}")
PYEOF

# Clean up temporary JSON file
rm -f schema.json schema.yaml

echo "✓ Generated types.go from schema.yaml"
