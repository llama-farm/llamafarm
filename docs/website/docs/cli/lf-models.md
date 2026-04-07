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

### `lf models path`

Query the local HuggingFace cache and emit a source→target transport plan for
each model configured in `llamafarm.yaml`. Designed for deployment tooling
(Ansible, Packer, Dockerfile builds) that needs to push model files onto a
target device without mirroring the entire HF cache.

The command is **query-only** — it never downloads or copies files. Run
`lf models pull` first to populate the cache, or pass `--ensure` to pull
missing models before emitting the plan.

```bash
# All models, tab-separated (default)
lf models path

# JSON with size + sha256 for every file
lf models path --format json

# Single model's weights file, shell-friendly
lf models path qwen3-1.7b --role weights --source-only

# Fresh build host: pull + emit in one shot
lf models path --format json --ensure
```

**Flags:**
- `--format json|tsv` — Output format (default: `tsv`)
- `--target-root <path>` — Base path used for computed `target` values; overrides `deployment.model_dir` in `llamafarm.yaml`
- `--role weights|mmproj|tokenizer|all` — Filter files by role (default: `all`)
- `--source-only` — Print only source paths, one per line
- `--ensure` — Run `lf models pull` for any missing models before emitting the plan

**JSON output shape:**
```json
{
  "target_root": "/opt/llamafarm/models",
  "manifest_target": "/opt/llamafarm/models/manifest.json",
  "models": [
    {
      "name": "qwen3-1.7b",
      "kind": "gguf",
      "quant": "Q4_K_M",
      "files": [
        {
          "role": "weights",
          "source": "/Users/me/.cache/huggingface/hub/models--unsloth--Qwen3-1.7B-GGUF/snapshots/abc123/qwen3-1.7b-Q4_K_M.gguf",
          "target": "/opt/llamafarm/models/qwen3-1.7b/model.Q4_K_M.gguf",
          "size": 1234567890,
          "sha256": "abc…"
        }
      ]
    }
  ]
}
```

**TSV output**: four tab-separated columns — `name`, `role`, `source`, `target` — one line per file. No `sha256` in TSV mode.

**Canonical target layout** (the shape the `target` paths describe):

```
<target-root>/
├── manifest.json                      ← downstream tooling writes this on the device
├── <alias>/
│   ├── model.<QUANT>.gguf             ← GGUF weights
│   └── mmproj.<precision>.gguf        ← optional multimodal projector
```

The CLI does not create this layout itself. It emits `target` paths that
follow this convention so that your Ansible playbook or Dockerfile can place
the files at the right spot. Format detection uses extension + GGUF magic
bytes — no `kind` metadata file is required.

**Target-root resolution precedence:**

1. `--target-root` flag
2. `deployment.model_dir` in `llamafarm.yaml`
3. Hardcoded default: `/opt/llamafarm/models`

**V1 scope:** `lf models path` currently supports GGUF models only. Non-GGUF
models (transformers, ultralytics) return a clear "not yet supported" error.

### Example: Ansible playbook

```yaml
- name: Populate HF cache on build host
  command: lf models pull
  delegate_to: localhost

- name: Get model transport plan
  command: lf models path --format json --target-root /opt/llamafarm/models
  delegate_to: localhost
  register: model_plan

- name: Create per-alias directories
  file:
    path: "{{ item | dirname }}"
    state: directory
    mode: "0755"
  loop: "{{ (model_plan.stdout | from_json).models | map(attribute='files') | flatten | map(attribute='target') | list }}"

- name: Push model files to device
  copy:
    src: "{{ item.source }}"
    dest: "{{ item.target }}"
    mode: "0644"
  loop: "{{ (model_plan.stdout | from_json).models | map(attribute='files') | flatten | list }}"
```

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
