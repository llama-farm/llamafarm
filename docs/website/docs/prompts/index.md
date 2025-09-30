---
title: Prompts
sidebar_position: 8
---

# Prompts

Prompts in LlamaFarm are simple but powerful: you define static instructions in `llamafarm.yaml`, and the runtime merges them with chat history and (optionally) RAG context. This section outlines current capabilities and roadmap plans.

## Prompt Configuration

```yaml
prompts:
  - role: system
    content: >-
      You are a regulatory assistant. Provide concise answers and cite sources by title.
  - role: user
    content: "Use bullet points by default."
```

- The array is preserved in order and prepended to each conversation.
- Roles should match what your provider understands (`system`, `user`, `assistant`).
- Combine with RAG by including instructions explaining how to use context snippets (the server injects them automatically).

## Best Practices

- **Explain context usage**: remind the model that context chunks contain citations or metadata.
- **Handle non-RAG scenarios**: mention what to do when no documents are retrieved (“answer from general knowledge” or “state that no information was found”).
- **Keep prompts concise**: long system instructions can reduce available tokens on smaller models.
- **Avoid conflicting instructions**: align prompts with agent handler expectations (structured vs. simple chat).

## Roadmap & Limitations

- Prompt templates, versioning, and evaluation tooling are in development. Track progress in the roadmap.
- For now, dynamic templating (Jinja, variables) is not built-in—generate prompts upstream if needed.

## Related Guides

- [Configuration Guide](../configuration/index.md)
- [RAG Guide](../rag/index.md) (for context injection tips)
- [Extending agent handlers](../extending/index.md#extend-runtimes)
