---
title: Prompts
sidebar_position: 8
---

# Prompts

Prompts in LlamaFarm are simple but powerful: you define instructions in `llamafarm.yaml`, and the runtime merges them with chat history and (optionally) RAG context. Prompts support **dynamic variable substitution** using Jinja2-style `{{variable}}` syntax, allowing you to customize behavior per-request.

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

## Dynamic Variables

Use Jinja2-style template syntax to inject values at request time. This is ideal for personalizing prompts per user, session, or context.

### Syntax

```yaml
# Required variable (error if not provided)
content: "Hello {{user_name}}"

# Variable with default (uses default if not provided)
content: "Welcome to {{company | Our Service}}"

# Whitespace is allowed
content: "Hello {{ user_name }}"
```

### Example: Personalized Customer Service

```yaml
prompts:
  - name: system
    messages:
      - role: system
        content: |
          You are a helpful customer service assistant for {{company_name | Acme Corp}}.
          You work in the {{department | General}} department.
          Current date: {{current_date | today}}
          Be professional, helpful, and concise in your responses.

  - name: context
    messages:
      - role: system
        content: |
          ## Customer Information
          - Name: {{user_name | Valued Customer}}
          - Account Tier: {{account_tier | standard}}
          - Preferred Language: {{language | English}}

          Adjust your responses based on the customer's tier:
          - basic: Focus on self-service options
          - standard: Provide helpful guidance
          - premium: Offer personalized, detailed assistance
```

### Passing Variables via API

Variables are passed in the `variables` field of the chat request:

```bash
curl -X POST http://localhost:14345/v1/projects/my-org/chatbot/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "variables": {
      "company_name": "TechCorp Solutions",
      "user_name": "Alice Johnson",
      "account_tier": "premium",
      "current_date": "2024-01-15"
    }
  }'
```

### Variable Behavior

| Scenario | Syntax | Result |
|----------|--------|--------|
| Variable provided | `{{name}}` with `{"name": "Alice"}` | `Alice` |
| Variable with default, not provided | `{{name \| Guest}}` with `{}` | `Guest` |
| Required variable missing | `{{name}}` with `{}` | **Error 400** |
| Null value | `{{name}}` with `{"name": null}` | Empty string |
| Non-string types | `{{count}}` with `{"count": 42}` | `42` (converted) |

### Supported Types

Variable values can be:
- **String** - Inserted as-is
- **Integer** - Converted to string (e.g., `42` → `"42"`)
- **Float** - Converted to string (e.g., `3.14` → `"3.14"`)
- **Boolean** - Converted to `True` or `False` (Python-style)
- **None/null** - Converted to empty string

Complex types (lists, dicts) are **not supported** and will raise an error.

## Best Practices

- **Use defaults for optional context**: Always provide defaults for variables that might not be passed in every request.
- **Keep required variables minimal**: Only mark variables as required (no default) when they're truly essential.
- **Explain context usage**: Remind the model that context chunks contain citations or metadata.
- **Handle non-RAG scenarios**: Mention what to do when no documents are retrieved.
- **Keep prompts concise**: Long system instructions can reduce available tokens on smaller models.
- **Avoid conflicting instructions**: Align prompts with agent handler expectations (structured vs. simple chat).

## Use Cases

Dynamic variables are ideal for:

- **Multi-tenant applications** - Customize branding per customer (`{{company_name}}`)
- **Personalization** - Inject user names, preferences, or roles
- **A/B testing** - Swap prompt variants without config changes
- **Context injection** - Pass session-specific data (dates, account tiers)
- **Internationalization** - Set language preferences per request

## Related Guides

- [Configuration Guide](../configuration/index.md) - Full variable syntax reference
- [API Reference](../api/index.md#send-chat-message-openai-compatible) - Using `variables` in requests
- [RAG Guide](../rag/index.md) - Context injection tips
- [Extending agent handlers](../extending/index.md#extend-runtimes)
