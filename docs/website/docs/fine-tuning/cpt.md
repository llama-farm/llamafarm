---
sidebar_label: CPT Training
title: Continued Pre-Training (CPT)
---

# Continued Pre-Training (CPT)

:::caution Alpha Feature
Fine-tuning is experimental. APIs may change between releases.
:::

CPT adapts a model to your domain using **unlabeled text** — no instruction/response pairs needed. Feed it documents, manuals, code, or any domain text and the model learns the language patterns and knowledge.

## When to Use CPT

- **Domain adaptation** — Make a general model understand your industry's terminology
- **Internal knowledge** — Train on company docs, wikis, runbooks, SOPs
- **Code repositories** — Adapt a model to your codebase style and patterns
- **Regulatory text** — FDA regulations, legal codes, compliance documents

## Training Data Format

CPT data is simple — just text:

```json
[
  {"text": "The FDA requires that all 510(k) submissions include a substantial equivalence determination..."},
  {"text": "LlamaFarm Universal Runtime supports GGUF models with automatic context window calculation..."},
  {"text": "When configuring Atmosphere mesh networking, the BLE transport layer handles discovery..."}
]
```

Each item should be a meaningful chunk of text (a paragraph, a document section, or a complete document). Very short snippets (< 50 tokens) are less effective.

## Examples

### Example 1: Company Knowledge Base (Qwen 8B)

Adapt a model to understand your product documentation:

```bash
curl -X POST http://localhost:11540/v1/finetune/cpt \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "dataset": [
      {"text": "LlamaFarm Architecture Overview\n\nLlamaFarm consists of three core services: the Main Server (port 14345), the Universal Runtime (port 11540), and the RAG service. The Main Server handles project management, agent orchestration, and API routing. The Universal Runtime loads and serves GGUF models with automatic memory management..."},
      {"text": "Vision Pipeline\n\nThe vision pipeline supports YOLO-based object detection and CLIP-based zero-shot classification. Models are stored in ~/.llamafarm/models/vision/ with versioned checkpoints. The cascade system tries models in order, escalating to more capable (or remote) models when confidence is low..."},
      {"text": "KV Cache System\n\nLlama-cache provides server-side KV cache management with tiered storage. When a conversation continues, the server reuses cached KV states instead of reprocessing the entire prompt. This reduces time-to-first-token by 15-27x for multi-turn conversations..."}
    ],
    "epochs": 1,
    "learning_rate": 5e-5,
    "max_seq_length": 4096
  }'
```

### Example 2: Regulatory Compliance (Gemma 4B)

Train on FDA or HIPAA documents:

```bash
curl -X POST http://localhost:11540/v1/finetune/cpt \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "dataset": [
      {"text": "21 CFR Part 820 — Quality System Regulation. Section 820.30 Design Controls. (a) Each manufacturer of any class III or class II device, and the class I devices listed in paragraph (a)(2) of this section, shall establish and maintain procedures to control the design of the device in order to ensure that specified design requirements are met..."},
      {"text": "HIPAA Security Rule — 45 CFR Part 164. The Security Rule requires covered entities to maintain reasonable and appropriate administrative, technical, and physical safeguards for protecting e-PHI. Specifically, covered entities must: ensure the confidentiality, integrity, and availability of all e-PHI they create, receive, maintain or transmit..."}
    ],
    "epochs": 1,
    "learning_rate": 5e-5,
    "embedding_learning_rate": 5e-6
  }'
```

## Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | *required* | HuggingFace model name |
| `dataset` | *required* | Training data (`[{{"text": "..."}}, ...]`) |
| `epochs` | `1` | Number of epochs (CPT typically needs only 1) |
| `batch_size` | `2` | Batch size |
| `learning_rate` | `5e-5` | Learning rate (lower than SFT — preserving existing knowledge) |
| `embedding_learning_rate` | `5e-6` | Separate LR for embedding layers |
| `lora_rank` | `16` | LoRA rank |
| `max_seq_length` | `2048` | Maximum sequence length |
| `max_steps` | `None` | Override epochs with fixed step count |
| `output_gguf` | `true` | Export to GGUF |
| `quantization` | `q8_0` | GGUF quantization method |

## Tips

- **Lower learning rate than SFT**: CPT uses `5e-5` (vs `2e-4` for SFT) to avoid catastrophic forgetting
- **1 epoch is usually enough**: Unlike SFT, CPT benefits from seeing data once with good coverage rather than repeating
- **Chunk your documents**: Split long documents into 512–2048 token chunks for better training
- **Embedding learning rate**: The separate `embedding_learning_rate` (default `5e-6`) allows fine-grained control over how much the token embeddings shift — important for domain-specific vocabulary
- **Combine with SFT**: The most effective pipeline is CPT first (domain knowledge) → SFT second (task behavior)

## CPT → SFT Pipeline

For best results, do domain adaptation first, then task-specific training:

```bash
# Step 1: CPT — teach the model your domain
curl -X POST http://localhost:11540/v1/finetune/cpt \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "dataset": [... your domain documents ...],
    "epochs": 1
  }'
# → produces adapter in ~/.llamafarm/models/llm/\{cpt_job_id\}/

# Step 2: SFT — teach it how to respond
# (use the CPT-adapted model as base for SFT)
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "dataset": [... your Q&A pairs ...],
    "epochs": 3
  }'
```

:::note
The CPT → SFT pipeline currently requires merging the CPT adapter back into the base model before SFT. Automatic adapter chaining is planned for a future release.
:::
