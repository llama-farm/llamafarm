# JSONSchema Generation and Validation Checklist

JSONSchema patterns for LlamaFarm configuration validation.

---

## Category: Schema Structure

### Use Draft-07 Schema Version

**What to check**: Schema declares JSON Schema draft-07

**Good pattern**:
```yaml
# yaml-language-server: $schema=http://json-schema.org/draft-07/schema#
$schema: http://json-schema.org/draft-07/schema#
title: LlamaFarm Config
type: object
```

**Why**: Draft-07 is well-supported by validators and code generators

**Search pattern**:
```bash
rg '\$schema.*draft-07' config/
```

**Severity**: Medium

---

### Declare Required Fields Explicitly

**What to check**: Required fields are listed in `required` array

**Good pattern**:
```yaml
type: object
required:
  - version
  - name
  - namespace
  - runtime
properties:
  version:
    type: string
  name:
    type: string
```

**Bad pattern**:
```yaml
properties:
  version:
    type: string
  # No 'required' array - all fields optional
```

**Search pattern**:
```bash
rg "^required:" config/schema.yaml
```

**Severity**: High

---

### Use Descriptive Property Descriptions

**What to check**: All properties have `description` field

**Good pattern**:
```yaml
properties:
  name:
    type: string
    description: Project name
    example: my-project
  namespace:
    type: string
    description: Project namespace
    example: my-namespace
```

**Why**: Descriptions appear in generated docs and IDE tooltips

**Severity**: Low

---

## Category: Type Definitions

### Use definitions for Reusable Types

**What to check**: Shared types are defined in `definitions` section

**Good pattern**:
```yaml
definitions:
  Tool:
    type: object
    required:
      - type
      - name
      - description
      - parameters
    properties:
      type:
        type: string
        enum: [function]
      name:
        type: string
      description:
        type: string
      parameters:
        type: object

# Usage via $ref
properties:
  tools:
    type: array
    items:
      $ref: "#/definitions/Tool"
```

**Search pattern**:
```bash
rg "definitions:" config/schema.yaml
```

**Severity**: Medium

---

### Use $ref for Cross-File References

**What to check**: Large schemas are split across files with `$ref`

**Good pattern**:
```yaml
# schema.yaml
properties:
  rag:
    $ref: "../rag/schema.yaml"
```

**Why**: Keeps schemas maintainable, allows domain-specific ownership

**Search pattern**:
```bash
rg '\$ref:' config/schema.yaml
```

**Severity**: Low

---

### Use enum for Fixed Value Sets

**What to check**: Fields with fixed options use `enum`

**Good pattern**:
```yaml
provider:
  type: string
  enum: [openai, ollama, lemonade, universal]
  description: Runtime provider for this model

transport:
  type: string
  enum: [stdio, http, sse]
  description: Connection transport to the MCP server
```

**Search pattern**:
```bash
rg "enum:" config/schema.yaml
```

**Severity**: Medium

---

## Category: String Constraints

### Use pattern for String Formats

**What to check**: Identifier fields use regex patterns

**Good pattern**:
```yaml
name:
  type: string
  pattern: "^[a-z][a-z0-9_]*$"
  description: Unique prompt set identifier

file_extensions:
  type: array
  items:
    type: string
    pattern: "^\\.[a-zA-Z0-9]+$"
```

**Why**: Catches invalid identifiers at validation time

**Search pattern**:
```bash
rg "pattern:" config/schema.yaml
```

**Severity**: Medium

---

### Use minLength and maxLength for Bounds

**What to check**: String fields have reasonable length limits

**Good pattern**:
```yaml
name:
  type: string
  pattern: "^[a-z][a-z0-9_]*$"
  minLength: 1
  maxLength: 50

description:
  type: string
  minLength: 10
  maxLength: 500
```

**Why**: Prevents empty strings and excessively long values

**Severity**: Low

---

## Category: Numeric Constraints

### Use minimum and maximum for Integer Bounds

**What to check**: Integer fields have appropriate bounds

**Good pattern**:
```yaml
priority:
  type: integer
  minimum: 0
  maximum: 1000
  default: 50
  description: Parser priority (lower = try first)

max_length:
  type: integer
  minimum: 1
  description: Maximum sequence length for tokenization
```

**Search pattern**:
```bash
rg "minimum:|maximum:" config/schema.yaml
```

**Severity**: Medium

---

## Category: Array Constraints

### Use minItems for Required Arrays

**What to check**: Arrays that must have items use `minItems`

**Good pattern**:
```yaml
databases:
  type: array
  description: Database definitions
  minItems: 1
  items:
    $ref: "#/definitions/Database"

parsers:
  type: array
  description: Document parsers in processing order
  minItems: 1
  items:
    $ref: "#/definitions/Parser"
```

**Why**: Ensures arrays have at least one element when required

**Search pattern**:
```bash
rg "minItems:" config/schema.yaml
```

**Severity**: Medium

---

### Use default for Optional Arrays

**What to check**: Optional arrays have sensible defaults

**Good pattern**:
```yaml
prompts:
  type: array
  description: List of named prompt sets
  items:
    $ref: "#/definitions/PromptSet"
  default: []

tools:
  type: array
  description: List of tools to use
  items:
    $ref: "#/definitions/Tool"
  default: []
```

**Severity**: Low

---

## Category: Object Constraints

### Use additionalProperties Appropriately

**What to check**: Objects specify whether extra properties are allowed

**Good pattern** - strict objects:
```yaml
type: object
additionalProperties: false
properties:
  name:
    type: string
```

**Good pattern** - extensible objects:
```yaml
model_api_parameters:
  type: object
  description: Additional parameters passed to the API
  additionalProperties: true
```

**Search pattern**:
```bash
rg "additionalProperties:" config/schema.yaml
```

**Severity**: Medium

---

## Category: Schema Compilation

### Dereference All $ref Before Validation

**What to check**: Schema is fully dereferenced before use

**Good pattern** (from compile_schema.py):
```python
import jsonref

def load_and_deref_schema(path: Path):
    """Load YAML schema and dereference all $refs."""
    with path.open(encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    deref = jsonref.JsonRef.replace_refs(
        schema,
        base_uri=path.as_uri(),
        loader=yaml_json_loader,
    )
    return jsonref_to_dict(deref, is_root=True)
```

**Why**: jsonschema validator doesn't resolve external $ref automatically

**Severity**: High

---

### Strip $schema and $id from Nested Refs

**What to check**: Dereferenced schema removes metadata from inlined refs

**Good pattern** (from compile_schema.py):
```python
def jsonref_to_dict(obj, is_root=False):
    """Convert jsonref proxies to plain dicts, stripping nested metadata."""
    if isinstance(obj, dict):
        if not is_root:
            schema_metadata_fields = {"$schema", "$id"}
            obj = {k: v for k, v in obj.items() if k not in schema_metadata_fields}
        return {k: jsonref_to_dict(v, is_root=False) for k, v in obj.items()}
```

**Why**: Nested $schema fields can confuse validators

**Severity**: Medium

---

### Validate Dereferenced Schema Output

**What to check**: Schema compilation validates the result

**Good pattern** (from compile_schema.py):
```python
if deref is None:
    raise ValueError("Schema dereferencing produced None")

if "type" not in deref and "properties" not in deref:
    raise ValueError("Schema is missing required top-level fields")

file_size = output_file.stat().st_size
if file_size < 100:
    raise ValueError(f"Schema file is suspiciously small ({file_size} bytes)")
```

**Severity**: High

---

## Category: Code Generation

### Use datamodel-codegen for Type Generation

**What to check**: Pydantic models are generated from schema

**Good pattern** (from generate_types.py):
```python
run_command([
    "uv", "run", "datamodel-codegen",
    "--input", "schema.deref.yaml",
    "--output", "datamodel.py",
    "--input-file-type=jsonschema",
    "--output-model-type=pydantic_v2.BaseModel",
    "--target-python-version=3.12",
    "--use-standard-collections",
    "--use-title-as-name",
    "--formatters=ruff-format",
    "--class-name=LlamaFarmConfig",
], cwd=config_dir)
```

**Why**: Ensures Pydantic models match schema exactly

**Severity**: High

---

### Use --use-standard-collections for Modern Types

**What to check**: Generated code uses `list[T]` not `List[T]`

**Good pattern**:
```bash
datamodel-codegen --use-standard-collections
```

**Result**:
```python
# Generated with modern syntax
messages: list[PromptMessage] = Field(...)
config: dict[str, Any] | None = Field(None, ...)
```

**Severity**: Low

---

### Use --use-title-as-name for Clear Class Names

**What to check**: Generated classes use schema `title` as class name

**Good pattern**:
```yaml
# In schema.yaml
title: PromptSet
type: object
```

```python
# Generated class
class PromptSet(BaseModel):
    ...
```

**Severity**: Low

---

## Category: Validation

### Use jsonschema for Runtime Validation

**What to check**: Config validation uses jsonschema library

**Good pattern** (from helpers/loader.py):
```python
import jsonschema

def _validate_config(config: dict, schema: dict) -> None:
    """Validate configuration against JSON schema."""
    try:
        jsonschema.validate(config, schema)
    except jsonschema.ValidationError as e:
        path_str = ".".join(str(p) for p in e.path)
        raise ConfigError(
            f"Configuration validation error: {e.message}"
            + (f" at path {path_str}" if path_str else "")
        ) from e
```

**Severity**: High

---

### Add Custom Validators for Complex Constraints

**What to check**: Constraints beyond JSONSchema are in `validators.py`

**What JSONSchema draft-07 CANNOT express**:
- Uniqueness of properties within arrays
- Cross-field references (e.g., prompt name must exist)
- Case-insensitive uniqueness
- Complex conditional validation

**Good pattern** (from validators.py):
```python
def validate_llamafarm_config(config_dict: dict[str, Any]) -> None:
    """Validate constraints beyond JSONSchema capabilities."""

    # Check unique prompt names
    prompt_names = [p.get("name") for p in config_dict.get("prompts", [])]
    duplicates = [name for name in prompt_names if prompt_names.count(name) > 1]
    if duplicates:
        raise ValueError(f"Duplicate prompt set names: {', '.join(set(duplicates))}")

    # Check model.prompts reference existing prompt sets
    prompt_names_set = {p.get("name") for p in config_dict.get("prompts", [])}
    for model in config_dict.get("runtime", {}).get("models", []):
        for prompt_ref in model.get("prompts", []):
            if prompt_ref not in prompt_names_set:
                raise ValueError(f"Model references non-existent prompt: {prompt_ref}")
```

**Severity**: High

---

### Provide Clear Validation Error Messages

**What to check**: Validation errors include path and context

**Good pattern**:
```python
raise ValueError(
    f"Model '{model_name}' references non-existent prompt set '{prompt_ref}'. "
    f"Available prompt sets: {', '.join(sorted(prompt_names_set))}"
)
```

**Bad pattern**:
```python
raise ValueError("Invalid prompt reference")  # No context
```

**Severity**: Medium

---

## Category: YAML Processing

### Use ruamel.yaml for Comment Preservation

**What to check**: YAML read/write preserves comments

**Good pattern** (from helpers/loader.py):
```python
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

def _get_ruamel_yaml() -> YAML:
    yaml_instance = YAML()
    yaml_instance.preserve_quotes = True
    yaml_instance.indent(mapping=2, sequence=4, offset=2)
    return yaml_instance
```

**Why**: Users add comments to configs; preserving them improves UX

**Severity**: Medium

---

### Convert Between CommentedMap and dict

**What to check**: Internal processing uses plain dicts

**Good pattern**:
```python
def _commented_map_to_dict(obj: Any) -> Any:
    """Recursively convert CommentedMap/CommentedSeq to plain dict/list."""
    if isinstance(obj, CommentedMap):
        return {k: _commented_map_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, CommentedSeq):
        return [_commented_map_to_dict(item) for item in obj]
    return obj
```

**Why**: Pydantic and jsonschema work with plain dicts

**Severity**: Medium

---

### Use LiteralScalarString for Multiline Strings

**What to check**: Multiline strings use YAML block scalar style

**Good pattern**:
```python
from ruamel.yaml.scalarstring import LiteralScalarString

def _dict_to_commented_map(obj: Any) -> Any:
    if isinstance(obj, str) and "\n" in obj:
        return LiteralScalarString(obj)
    return obj
```

**Result in YAML**:
```yaml
content: |
  This is a multiline
  string that preserves
  formatting nicely.
```

**Severity**: Low
