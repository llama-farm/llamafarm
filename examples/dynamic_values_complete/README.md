# Dynamic Values Complete Example

This example demonstrates the full capabilities of LlamaFarm's dynamic value substitution feature.

## Features Demonstrated

1. **Dynamic Prompts** - System prompts with `{{variable}}` placeholders
2. **Dynamic Tools** - Tool descriptions and parameters with variables
3. **Default Values** - `{{variable | default}}` syntax for optional variables
4. **Per-Request Variables** - Different values for each API call
5. **Backwards Compatibility** - Works alongside existing static configs

## Configuration

The `llamafarm.yaml` file defines:

- **Two prompt sets** with dynamic customer context
- **Two tools** with dynamic company/department references
- **Multiple variables** with sensible defaults

## Variables Used

| Variable | Default | Description |
|----------|---------|-------------|
| `company_name` | Acme Corp | Company name in prompts/tools |
| `department` | General | Department name |
| `user_name` | Valued Customer | Customer's name |
| `account_tier` | standard | Customer tier (basic/standard/premium) |
| `current_date` | today | Current date string |
| `language` | English | Preferred language |

## Running the Demo

```bash
# Make sure server is running
nx start server  # or however you start LlamaFarm

# Run the demo
bash examples/dynamic_values_complete/demo.sh
```

## API Usage

```bash
curl -X POST http://localhost:8000/v1/examples/dynamic_values_complete/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "variables": {
      "company_name": "My Company",
      "user_name": "Alice",
      "account_tier": "premium"
    }
  }'
```

## Template Syntax

```yaml
# Required variable (error if not provided)
content: "Hello {{user_name}}"

# Variable with default (uses default if not provided)
content: "Welcome to {{company | Our Service}}"

# Whitespace is allowed
content: "Hello {{ user_name }}"
```
