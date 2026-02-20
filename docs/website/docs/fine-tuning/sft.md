---
sidebar_label: SFT Training
title: Supervised Fine-Tuning (SFT)
---

# Supervised Fine-Tuning (SFT)

:::caution Alpha Feature
Fine-tuning is experimental. APIs may change between releases.
:::

SFT trains a model on labeled input/output pairs — the most common way to customize an LLM for a specific task.

## When to Use SFT

- **Custom assistants** — Train a model to respond in your style, tone, or domain
- **Tool calling** — Teach a model to use your specific APIs and functions
- **Classification** — Fine-tune for structured output (JSON, categories, labels)
- **Domain expertise** — Medical, legal, financial Q&A with vetted responses

## Training Data Formats

LlamaFarm auto-detects your dataset format. You can use any of these:

### ShareGPT Format (Recommended)

The most flexible format — supports multi-turn conversations with system prompts.

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful medical assistant."},
      {"from": "human", "value": "What are the symptoms of type 2 diabetes?"},
      {"from": "gpt", "value": "Common symptoms include increased thirst, frequent urination, blurred vision, and fatigue."}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "Explain insulin resistance."},
      {"from": "gpt", "value": "Insulin resistance occurs when cells in your body don't respond well to insulin..."}
    ]
  }
]
```

### Chat Format (OpenAI-style)

If you already have data in OpenAI's chat format:

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a code reviewer."},
      {"role": "user", "content": "Review this function: def add(a, b): return a + b"},
      {"role": "assistant", "content": "The function is simple and correct. Consider adding type hints: def add(a: int, b: int) -> int"}
    ]
  }
]
```

### Alpaca Format

Simple instruction/input/output triples:

```json
[
  {
    "instruction": "Summarize the following text.",
    "input": "LlamaFarm is a local AI infrastructure platform that lets you run models on your own hardware...",
    "output": "LlamaFarm is a self-hosted AI platform for running local models."
  }
]
```

## Examples

### Example 1: Customer Support Bot (Gemma 4B)

Train a small model to handle support tickets in your company's voice:

```bash
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "dataset": [
      {
        "conversations": [
          {"from": "system", "value": "You are Acme Corp support. Be friendly, concise, and always offer to escalate."},
          {"from": "human", "value": "My order hasnt arrived yet"},
          {"from": "gpt", "value": "I am sorry to hear that! Let me look into your order. Could you share your order number? If this is urgent, I can escalate to our shipping team right away."}
        ]
      }
    ],
    "epochs": 5,
    "lora_rank": 16,
    "learning_rate": 2e-4,
    "max_seq_length": 2048
  }'
```

### Example 2: Tool-Calling Agent (FunctionGemma 270M)

Train a tiny model to route function calls — fast inference on edge devices:

```bash
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/FunctionGemma-270M",
    "dataset": [
      {
        "conversations": [
          {"from": "human", "value": "Turn on the living room lights"},
          {"from": "gpt", "value": "{\"function\": \"smart_home.lights\", \"args\": {\"room\": \"living_room\", \"state\": \"on\"}}"}
        ]
      },
      {
        "conversations": [
          {"from": "human", "value": "What is the temperature outside?"},
          {"from": "gpt", "value": "{\"function\": \"weather.current\", \"args\": {\"metric\": \"temperature\"}}"}
        ]
      }
    ],
    "epochs": 10,
    "lora_rank": 8,
    "batch_size": 4,
    "train_on_responses_only": true
  }'
```

### Example 3: Medical Q&A (Llama 3.2 3B)

Fine-tune for domain-specific question answering:

```bash
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "chat_template": "llama-3",
    "dataset": [
      {
        "conversations": [
          {"from": "system", "value": "You are a medical information assistant. Always recommend consulting a healthcare provider."},
          {"from": "human", "value": "What is the standard dosage for metformin?"},
          {"from": "gpt", "value": "Metformin is typically started at 500mg once or twice daily, with gradual increases up to 2000mg/day based on tolerance and blood glucose response. However, dosing should always be determined by a prescribing physician based on individual factors."}
        ]
      }
    ],
    "epochs": 3,
    "learning_rate": 1e-4,
    "max_seq_length": 4096
  }'
```

## Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `model` | *required* | HuggingFace model name (e.g., `google/gemma-3-4b-it`) |
| `dataset` | *required* | Training data (inline JSON array) |
| `dataset_format` | `auto` | `sharegpt`, `chat`, `alpaca`, or `auto` |
| `chat_template` | `auto` | `llama-3`, `chatml`, `gemma`, `qwen`, `phi`, `mistral` |
| `train_on_responses_only` | `true` | Mask user turns in loss calculation |
| `epochs` | `3` | Number of training epochs (1–100) |
| `batch_size` | `2` | Batch size (1–128) |
| `learning_rate` | `2e-4` | Learning rate |
| `lora_rank` | `16` | LoRA rank (1–256). Higher = more capacity, more memory |
| `lora_alpha` | `16` | LoRA alpha scaling factor |
| `max_seq_length` | `2048` | Maximum sequence length (128–32768) |
| `max_steps` | `None` | Override epochs with fixed step count |
| `warmup_steps` | `10` | Learning rate warmup steps |
| `gradient_accumulation_steps` | `2` | Gradient accumulation (effective batch = batch_size × this) |
| `output_gguf` | `true` | Export trained model to GGUF format |
| `quantization` | `q8_0` | GGUF quantization method |

## Tips

- **Start small**: Use a 1B or 270M model first to validate your data pipeline before training larger models
- **`train_on_responses_only: true`** (default): The model only learns from assistant responses, not user prompts — this is almost always what you want
- **More data > more epochs**: 500 diverse examples for 3 epochs beats 50 examples for 30 epochs
- **LoRA rank 16** is a good default. Use 8 for tiny models, 32–64 for complex tasks
- **Learning rate**: `2e-4` works well for most SFT tasks. Lower (`1e-4`) for larger models
- **Monitor training loss**: If loss plateaus early, you may need more diverse data, not more epochs

## Output

Trained models are saved to `~/.llamafarm/models/llm/{job_id}/`:

```
~/.llamafarm/models/llm/{job_id}/
├── lora_adapter/          # LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapters.safetensors
│   └── tokenizer files...
├── gguf/                  # GGUF export (if enabled)
│   └── model-q8_0.gguf
├── config.json            # Training configuration
├── training.log           # Training output log
├── training_log.jsonl     # Step-by-step metrics
└── result.json            # Final results
```
