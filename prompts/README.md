# LlamaFarm Prompts

A simple, JSON-Schema compliant prompt configuration system in YAML format.

## Structure

```
prompts/
├── schema.yaml           # JSON-Schema definition for prompt configurations
├── default-prompts.yaml  # Default prompts with all types (no specific use-case)
└── examples/            # Use-case specific examples
    ├── customer-service.yaml  # Customer service prompts
    ├── code-review.yaml      # Code review and analysis prompts
    └── data-analysis.yaml    # Data analysis prompts
```

## Schema Overview

The schema defines two main arrays:

### 1. Global Prompts
System-wide prompts that apply across all interactions:
- Base system instructions
- Safety guidelines
- Context awareness
- Domain expertise

### 2. Prompts
Specific prompts for different purposes with types:
- `system` - System-level instructions
- `user` - User input templates
- `assistant` - Assistant response templates
- `function` - Function/API call templates
- `tool` - External tool usage templates
- `example` - Example demonstrations
- `instruction` - Step-by-step instructions
- `context` - Context provision templates
- `query` - Structured query templates
- `response` - Formatted response templates

## Features

- **JSON-Schema Compliant**: Valid JSON-Schema in YAML format
- **Full Variable Support**: Each prompt includes variable definitions with:
  - Type validation (string, number, boolean, array, object)
  - Required/optional flags
  - Default values
  - Validation rules (min/max, patterns, enums)
  - Descriptions
- **Examples**: Each prompt can include usage examples
- **Metadata**: Rich metadata for organization and discovery
- **Variations**: Multiple similar prompts (simple_1, simple_2, etc.) show different approaches

## Usage

1. **Schema Validation**: Use the `schema.yaml` to validate prompt configurations
2. **Default Templates**: Start with `default-prompts.yaml` for general use
3. **Customize for Use Case**: Adapt examples for specific domains:
   - Customer service
   - Code review
   - Data analysis
   - (Add your own)

## File Format

Each prompt configuration file contains:

```yaml
version: "1.0.0"

metadata:
  name: Configuration name
  description: What these prompts are for
  author: Author name
  tags: [tag1, tag2]

global_prompts:
  - id: unique_id
    name: Human-readable name
    content: Prompt template with {{variables}}
    variables:
      - name: variable_name
        type: string
        required: true
        description: What this variable does

prompts:
  - id: unique_id
    name: Human-readable name
    type: system|user|assistant|...
    content: Prompt template
    variables: [...]
    examples: [...]
```

## Variable Substitution

Variables use Handlebars-style syntax:
- `{{variable}}` - Simple substitution
- `{{#if variable}}...{{/if}}` - Conditional blocks
- `{{#each array}}...{{/each}}` - Iteration

## Examples

See the `examples/` directory for complete, working examples for different use cases.

## MVP Status

This is a simplified MVP focusing on:
- Clear schema definition
- Comprehensive prompt types
- Full variable support
- Use-case examples

Implementation of the runtime system will follow in future iterations.