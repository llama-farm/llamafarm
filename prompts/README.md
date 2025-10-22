# Prompts

Simple helpers for working with the `prompts` array in `llamafarm.yaml`.

## Current Model
- Prompts are defined as named sets in project config: each has a `name` and a list of `messages`.
- Each message has a `role` and `content`.
- The server/CLI prepend these messages before user input (and before injected RAG context).
- Models can select which prompt sets to use via `prompts: [list of names]`; if omitted, all prompts stack in definition order.
- There is no separate prompt templating system; generate dynamic content upstream if required.

## Files
- `schema.yaml` – legacy prompt schema (kept for reference, not enforced by the main config).
- `default-prompts.yaml` / `examples/` – sample prompt collections you can adapt manually.

## Recommended Usage
Configure prompts in `llamafarm.yaml`:
```yaml
prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a regulatory assistant. Cite document titles when possible.
```

See the [Prompts doc](../docs/website/docs/prompts/index.md) for best practices.
