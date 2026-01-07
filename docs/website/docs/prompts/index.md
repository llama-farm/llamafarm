---
title: Prompts
sidebar_position: 8
---

# Prompts

Prompts in LlamaFarm are simple but powerful: you define instructions in `llamafarm.yaml`, and the runtime merges them with chat history and (optionally) RAG context. Prompts support **template variables** for dynamic customization.

## Prompt Configuration

```yaml
prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a regulatory assistant. Provide concise answers and cite sources by title.
      - role: user
        content: "Use bullet points by default."
```

- Prompts are named sets that can be selectively applied to models.
- Messages within each prompt set are preserved in order and prepended to conversations.
- Roles should match what your provider understands (`system`, `user`, `assistant`).
- Models can specify which prompt sets to use via `prompts: [list of names]`; if omitted, all prompts stack in definition order.
- Combine with RAG by including instructions explaining how to use context snippets (the server injects them automatically).

## Prompt Variables

Use `{{variable_name}}` syntax to create dynamic prompts. Variables are substituted before the prompt is sent to the model.

### Example: Persona Prompt

```yaml
prompts:
  - name: persona
    messages:
      - role: system
        content: |
          You are {{persona_name}}, a {{persona_role}}.
          Your expertise is in {{expertise}}.
          Respond in a {{tone}} tone.
          Keep responses {{response_style}}.

runtime:
  models:
    - name: analyst
      provider: ollama
      model: qwen3:8b
      prompts: [persona]

      # Default values for the variables above
      variables:
        persona_name: "DataBot"
        persona_role: "senior data analyst"
        expertise: "statistical analysis and business intelligence"
        tone: "professional"
        response_style: "concise and data-driven"

    - name: casual
      provider: ollama
      model: qwen3:8b
      prompts: [persona]

      # Same prompt, different personality
      variables:
        persona_name: "Buddy"
        persona_role: "helpful friend"
        expertise: "general knowledge"
        tone: "casual and friendly"
        response_style: "warm and approachable"
```

### Variable Resolution

**Priority:** API Request > Model Defaults > Empty String

1. Variables provided in the API request take precedence
2. If not in the request, model-level `variables` defaults are used
3. If no default exists, the variable becomes an empty string

### Override at Request Time

```bash
curl -X POST http://localhost:8000/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Introduce yourself"}],
    "model": "analyst",
    "variables": {
      "tone": "casual",
      "persona_name": "CustomBot"
    }
  }'
```

This overrides `tone` and `persona_name` while keeping other variables from the model defaults.

### Variable Naming Rules

- Use word characters only: letters, numbers, and underscores
- Case-sensitive: `{{Name}}` and `{{name}}` are different variables
- Examples: `{{persona_name}}`, `{{STYLE}}`, `{{version_1}}`

## Best Practices

- **Explain context usage**: remind the model that context chunks contain citations or metadata.
- **Handle non-RAG scenarios**: mention what to do when no documents are retrieved (“answer from general knowledge” or “state that no information was found”).
- **Keep prompts concise**: long system instructions can reduce available tokens on smaller models.
- **Avoid conflicting instructions**: align prompts with agent handler expectations (structured vs. simple chat).

## Roadmap & Limitations

- Prompt versioning and evaluation tooling are in development. Track progress in the roadmap.
- Variable templating uses simple `{{variable}}` syntax. For advanced templating (conditionals, loops), generate prompts upstream.

## Related Guides

- [Configuration Guide](../configuration/index.md)
- [RAG Guide](../rag/index.md) (for context injection tips)
- [Extending agent handlers](../extending/index.md#extend-runtimes)
