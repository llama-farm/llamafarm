---
name: lf-config-validator
description: Validate LlamaFarm configuration files (llamafarm.yaml). Use when creating, editing, or troubleshooting LlamaFarm config files. Provides knowledge about config schema structure, validation rules, common errors and how to fix them.
allowed-tools: Bash, Read, Edit, Grep, Glob
---

# LlamaFarm Config Validator

This skill provides guidance for validating LlamaFarm configuration files and fixing validation errors.

## How to Validate

Run the validation script from the `config/` directory:

```bash
cd config && uv run python validate_config.py <config_path>
cd config && uv run python validate_config.py <config_path> --verbose
```

**Exit codes:**
- `0` - Valid configuration
- `1` - Invalid configuration (schema or validation error)
- `2` - File not found or other error

**Example:**
```bash
cd config && uv run python validate_config.py ../llamafarm.yaml --verbose
```

## Validation Layers

LlamaFarm configs are validated at three levels:

### 1. JSON Schema Validation
Validates structure, types, required fields, and enums against `config/schema.yaml` and `rag/schema.yaml`.

### 2. Custom Validators (`config/validators.py`)
Business rules beyond JSON Schema:
- Unique prompt set names
- Unique dataset names (case-insensitive)
- Dataset naming patterns (`[a-zA-Z0-9_-]+`, max 100 chars)
- Model-to-prompt references (prompts must exist)

### 3. Pydantic Model Validation
Type coercion and defaults via auto-generated `config/datamodel.py`.

---

## Common Error Types & Fixes

### Parse Errors

#### Malformed YAML Syntax

**Error:** `Error loading YAML file /path/to/file.yaml: while scanning...`

**Fix:** Check for:
- Incorrect indentation (YAML uses spaces, not tabs)
- Missing colons after keys
- Unquoted special characters
- Mismatched brackets/braces

```yaml
# Wrong - tabs used for indentation
runtime:
	models: []

# Correct - spaces used for indentation
runtime:
  models: []
```

---

### Schema Errors

#### Missing Required Fields

**Error:** `'version' is a required property`

**Fix:** Add the missing field at the config root:
```yaml
version: v1
name: my-project
namespace: default
runtime:
  models: []
```

**Required top-level fields:**
- `version` (must be `v1`)
- `name` (project name)
- `namespace` (project namespace)
- `runtime` (runtime configuration object)

---

#### Wrong Field Type

**Error:** `'models' is not of type 'array'`

**Fix:** Convert to correct type:
```yaml
# Wrong
runtime:
  models: "my-model"

# Correct
runtime:
  models:
    - name: my-model
      provider: openai
      model: gpt-4
```

---

#### Invalid Enum Value

**Error:** `'v2' is not one of ['v1']`

**Fix:** Use a valid enum value:
```yaml
# Wrong
version: v2

# Correct
version: v1
```

**Common enums:**
- `version`: `v1`
- `provider`: `openai`, `ollama`, `lemonade`, `universal`
- `mcp.transport`: `stdio`, `http`, `sse`

---

#### Unknown Property (Typo)

**Error:** `Additional properties are not allowed ('modles' was unexpected)`

**Fix:** Check spelling and remove unknown properties:
```yaml
# Wrong
runtime:
  modles: []

# Correct
runtime:
  models: []
```

---

### Custom Validation Errors

#### Duplicate Prompt Names

**Error:** `Duplicate prompt set names found: default, default. Each prompt set must have a unique name.`

**Fix:** Ensure each prompt set has a unique name:
```yaml
prompts:
  - name: default
    messages: [...]
  - name: advanced  # Changed from 'default'
    messages: [...]
```

---

#### Duplicate Dataset Names

**Error:** `Duplicate dataset names found: 'my_data'. Each dataset must have a unique name (case-insensitive).`

**Fix:** Use unique names (comparison is case-insensitive):
```yaml
datasets:
  - name: my_data
    database: db1
    data_processing_strategy: default
  - name: my_data_v2  # Changed from 'My_Data'
    database: db2
    data_processing_strategy: default
```

---

#### Invalid Dataset Name Characters

**Error:** `Dataset name 'my data!' contains invalid characters. Dataset names can only contain letters, numbers, underscores (_), and hyphens (-).`

**Fix:** Use only `[a-zA-Z0-9_-]`:
```yaml
# Wrong
- name: "my data!"

# Correct
- name: my_data
- name: my-data-2
```

---

#### Dataset Name Too Long

**Error:** `Dataset name '...' is too long (max 100 characters). Please use a shorter name.`

**Fix:** Shorten the name to 100 characters or less.

---

#### Model References Non-Existent Prompt

**Error:** `Model 'my-model' references non-existent prompt set 'fancy'. Available prompt sets: default, system`

**Fix:** Either add the missing prompt set or fix the reference:
```yaml
prompts:
  - name: default
    messages: [...]
  - name: fancy  # Add the missing prompt set
    messages: [...]

runtime:
  models:
    - name: my-model
      provider: openai
      model: gpt-4
      prompts: ["fancy"]  # Now valid
```

---

### RAG Configuration Errors

#### Missing Database in Dataset

**Error:** `'database' is a required property` (at datasets.0)

**Fix:** Add the required database reference:
```yaml
datasets:
  - name: my_dataset
    database: my_database  # Required
    data_processing_strategy: default
```

---

#### Invalid RAG Database Type

**Error:** `'InvalidStore' is not one of ['ChromaStore', 'QdrantStore']`

**Fix:** Use a valid database type:
```yaml
rag:
  databases:
    - name: my_db
      type: ChromaStore  # Valid types: ChromaStore, QdrantStore
      config:
        collection_name: my_collection
```

---

## Config File Structure Reference

### Minimal Valid Config

```yaml
version: v1
name: my-project
namespace: default

runtime:
  models:
    - name: default
      provider: openai
      model: gpt-4
      base_url: https://api.openai.com/v1
```

### Full Config Structure

```yaml
version: v1           # Required: must be "v1"
name: string          # Required: project name
namespace: string     # Required: project namespace

prompts:              # Optional: list of prompt sets
  - name: string      # Required: unique identifier (pattern: ^[a-z][a-z0-9_]*$)
    messages:         # Required: list of messages
      - role: string  # Required: system/user/assistant/tool
        content: string

runtime:              # Required
  default_model: string  # Optional: name of default model
  models:             # List of model configs
    - name: string    # Required: unique identifier
      provider: enum  # Required: openai|ollama|lemonade|universal
      model: string   # Required: model name/ID
      base_url: string
      api_key: string
      prompts: [string]  # References to prompt set names
      # ... additional fields

datasets:             # Optional: list of datasets
  - name: string      # Required: [a-zA-Z0-9_-]+, max 100 chars
    database: string  # Required: RAG database name
    data_processing_strategy: string  # Required

rag:                  # Optional: RAG configuration
  default_database: string
  databases:
    - name: string
      type: enum      # ChromaStore|QdrantStore
      config: object
      embedding_strategies: [...]
      retrieval_strategies: [...]
  data_processing_strategies:
    - name: string
      parsers: [...]
      extractors: [...]

mcp:                  # Optional: MCP server configuration
  servers:
    - name: string
      transport: enum  # stdio|http|sse
      command: string  # For stdio
      base_url: string # For http
```

---

## Troubleshooting Guide

### Step 1: Run Validation
```bash
cd config && uv run python validate_config.py <path> --verbose
```

### Step 2: Identify Error Type
- **Schema error** → Missing field, wrong type, invalid value
- **Validation error** → Duplicates, bad references, naming issues

### Step 3: Locate the Problem
The error message includes the JSON path (e.g., `at path runtime.models.0.provider`).

### Step 4: Apply the Fix
Use the error type guides above to fix the specific issue.

### Step 5: Re-validate
Run validation again to confirm the fix and check for additional errors.

---

## Key Files

| File | Purpose |
|------|---------|
| `config/schema.yaml` | Main JSON Schema |
| `rag/schema.yaml` | RAG configuration schema |
| `config/validators.py` | Custom validation rules |
| `config/helpers/loader.py` | Config loading and validation |
| `config/validate_config.py` | CLI validation script |
| `config/tests/minimal_config.yaml` | Minimal valid config example |
| `config/tests/sample_config.yaml` | Full config example |
