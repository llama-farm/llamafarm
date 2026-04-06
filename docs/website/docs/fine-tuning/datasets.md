---
sidebar_label: Dataset Formats
title: Dataset Formats
---

# Dataset Formats

:::caution Alpha Feature
Fine-tuning is experimental. APIs may change between releases.
:::

LlamaFarm auto-detects your dataset format. Use the `/v1/finetune/validate` endpoint to check your data before training.

## SFT Formats

### ShareGPT

Multi-turn conversations with `from`/`value` pairs. Supports `human`, `gpt`, and `system` roles.

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful assistant."},
      {"from": "human", "value": "What is 2+2?"},
      {"from": "gpt", "value": "4"}
    ]
  }
]
```

**Auto-detected when:** Items have a `conversations` key containing a list of dicts with `from` and `value`.

### Chat (OpenAI-style)

Standard chat completion format with `role`/`content` pairs.

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ]
  }
]
```

**Auto-detected when:** Items have a `messages` key containing a list of dicts with `role` and `content`.

### Alpaca

Simple instruction/input/output triples. The `input` field is optional.

```json
[
  {
    "instruction": "Translate the following to French.",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  },
  {
    "instruction": "What is the capital of France?",
    "output": "Paris"
  }
]
```

**Auto-detected when:** Items have `instruction` and `output` keys.

## CPT Format

### Raw Text

Simple text documents for continued pre-training.

```json
[
  {"text": "Your first document or paragraph..."},
  {"text": "Another document or section..."},
  {"text": "More domain-specific content..."}
]
```

**Requirements:**
- Each item must have a `text` key
- Text should be meaningful chunks (50+ tokens recommended)
- No conversation structure needed

## Format Conversion

LlamaFarm can convert between formats automatically:

| From | To | Supported |
|---|---|---|
| Chat → ShareGPT | ✅ | Role mapping: `user`→`human`, `assistant`→`gpt` |
| ShareGPT → Chat | ✅ | Role mapping: `human`→`user`, `gpt`→`assistant` |
| Chat → Alpaca | ✅ | First user message → `instruction`, last assistant → `output` |
| Alpaca → Chat | ✅ | `instruction` → user message, `output` → assistant message |

## Validation

Always validate before training:

```bash
curl -X POST http://localhost:11540/v1/finetune/validate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [...your data...],
    "dataset_format": "auto"
  }'
```

The validator checks:
- Format detection and consistency
- Required fields present
- Role values are valid (`human`/`gpt`/`system` for ShareGPT, `user`/`assistant`/`system` for Chat)
- Dataset is non-empty (max 100K examples)
- Returns warnings for unusual patterns

## Dataset Size Guidelines

| Task | Minimum | Recommended | Notes |
|---|---|---|---|
| Tool calling | 50 | 200–500 | High-quality, diverse function calls |
| Q&A / Classification | 100 | 500–2000 | Cover all categories and edge cases |
| Custom assistant style | 200 | 1000–5000 | More data = more consistent tone |
| Domain CPT | 500 | 5000–50000 | Text chunks from your domain |

Quality matters more than quantity. 200 carefully curated examples often outperform 5000 noisy ones.
