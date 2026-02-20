---
sidebar_label: Fine-Tuning
title: Fine-Tuning (Alpha)
---

# Fine-Tuning

:::caution Alpha Feature
Fine-tuning is **experimental** and under active development. APIs may change between releases. Currently supported on **Apple Silicon (MLX)** and **NVIDIA GPU (CUDA)** only.
:::

LlamaFarm's Universal Runtime includes built-in LoRA fine-tuning powered by [Unsloth](https://github.com/unslothai/unsloth). Train custom models directly through the API — no separate training infrastructure needed.

## What You Can Do

- **Supervised Fine-Tuning (SFT)** — Train on instruction/response pairs, chat conversations, or tool-calling data
- **Continued Pre-Training (CPT)** — Adapt a base model to your domain with unlabeled text
- **LoRA adapters** — Efficient training that produces small adapter files (not full model copies)
- **GGUF export** — Automatically convert trained adapters to GGUF for local inference

## Supported Models

Fine-tuning works with any HuggingFace model supported by Unsloth. Tested models include:

| Model | Size | Use Case | Template |
|---|---|---|---|
| `google/gemma-3-4b-it` | 4B | General instruction following | `gemma` |
| `Qwen/Qwen3-8B` | 8B | Multilingual, reasoning | `qwen` |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Fast, efficient SFT | `llama-3` |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | European languages, code | `mistral` |
| `microsoft/Phi-4` | 14B | Code, math, reasoning | `phi` |
| `google/gemma-3-1b-it` | 1B | Edge deployment, rapid iteration | `gemma` |
| `unsloth/FunctionGemma-270M` | 270M | Tool/function calling | `gemma` |

:::info GGUF Models Cannot Be Fine-Tuned Directly
If you have a `.gguf` file, you need the original HuggingFace model. The API will detect GGUF references and suggest the base model.
:::

## Quick Start

### 1. Check the Runtime Has Fine-Tuning Support

```bash
curl http://localhost:11540/v1/finetune/templates
```

If fine-tuning is available, this returns the list of supported chat templates.

### 2. Validate Your Dataset

```bash
curl -X POST http://localhost:11540/v1/finetune/validate \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": [
      {"conversations": [
        {"from": "human", "value": "What is LlamaFarm?"},
        {"from": "gpt", "value": "LlamaFarm is a local AI infrastructure platform."}
      ]}
    ]
  }'
```

### 3. Start Training

```bash
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/FunctionGemma-270M",
    "dataset": [
      {"conversations": [
        {"from": "human", "value": "What is LlamaFarm?"},
        {"from": "gpt", "value": "LlamaFarm is a local AI infrastructure platform."}
      ]}
    ],
    "epochs": 3,
    "lora_rank": 16
  }'
```

### 4. Monitor Progress

```bash
curl http://localhost:11540/v1/finetune/jobs/{job_id}
```

## Next Steps

- [SFT Training Guide](./sft) — Detailed supervised fine-tuning walkthrough
- [CPT Training Guide](./cpt) — Domain adaptation with unlabeled text
- [API Reference](./api) — Full endpoint documentation
- [Dataset Formats](./datasets) — Supported data formats and examples
