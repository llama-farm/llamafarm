---
title: lf models
sidebar_position: 8
---

# `lf models`

Manage and interact with models configured in your project. The models command provides subcommands to list available models and switch between them during chat sessions.

## Synopsis

```
lf models list [namespace/project] [flags]
```

If you omit `namespace/project`, the CLI resolves them from `llamafarm.yaml`.

## Subcommands

### `lf models list`

List all models configured in your project with their descriptions and providers.

```bash
lf models list                    # List models from current project
lf models list company/project    # List models from specific project
```

**Output includes:**
- Model name (used for `--model` flag)
- Description
- Provider (ollama, lemonade, openai, etc.)
- Default status

## Using Models

After listing available models, use them in chat commands:

```bash
# Use a specific model
lf chat --model powerful "Complex reasoning question"

# Use the default model (no flag needed)
lf chat "Regular question"
```

## Multi-Model Configuration

Configure multiple models in `llamafarm.yaml`:

```yaml
runtime:
  default_model: fast

  models:
    fast:
      description: "Fast Ollama model"
      provider: ollama
      model: gemma3:1b

    powerful:
      description: "More capable model"
      provider: ollama
      model: qwen3:8b

    lemon:
      description: "Lemonade local model"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        backend: llamacpp
        port: 11534
```

## Examples

```bash
# List all models
lf models list

# Use a specific model for chat
lf chat --model lemon "What is the capital of France?"

# Compare responses from different models
lf chat --model fast "Quick answer needed"
lf chat --model powerful "Complex reasoning task"
```

## See Also

- [`lf chat`](./lf-chat.md)
- [Models & Runtime Guide](../models/index.md)
- [Configuration Guide](../configuration/index.md)
