# Fine-Tuning Addon for LlamaFarm Universal Runtime

Cross-platform fine-tuning addon powered by Unsloth. Supports both MLX (Apple Silicon) and CUDA backends.

## Features

- **Supervised Fine-Tuning (SFT)**: Fine-tune models on instruction/chat datasets
- **Continued Pre-Training (CPT)**: Continue training on raw text data
- **Cross-Platform**: Automatic backend detection (MLX on macOS, CUDA on Linux/Windows)
- **LoRA Training**: Efficient parameter-efficient fine-tuning
- **GGUF Export**: Automatically export trained models to GGUF format
- **Sequential Job Queue**: Process one training job at a time
- **OOM Isolation**: Training runs in subprocess to prevent memory issues

## Installation

### macOS (Apple Silicon)
```bash
pip install unsloth-mlx
```

### Linux/Windows (NVIDIA GPU)
```bash
pip install unsloth
```

## API Endpoints

### Create SFT Job
```
POST /v1/finetune/sft
```

Request body:
```json
{
  "model": "meta-llama/Llama-3-8B",
  "dataset": [
    {
      "conversations": [
        {"from": "human", "value": "Hello!"},
        {"from": "gpt", "value": "Hi there!"}
      ]
    }
  ],
  "epochs": 3,
  "batch_size": 2,
  "learning_rate": 2e-4,
  "output_gguf": true,
  "quantization": "q8_0"
}
```

### Create CPT Job
```
POST /v1/finetune/cpt
```

Request body:
```json
{
  "model": "meta-llama/Llama-3-8B",
  "dataset": [
    {"text": "This is training text..."}
  ],
  "epochs": 1,
  "learning_rate": 5e-5,
  "embedding_learning_rate": 5e-6
}
```

### List Jobs
```
GET /v1/finetune/jobs
```

### Get Job Status
```
GET /v1/finetune/jobs/{job_id}
```

### Cancel Job
```
DELETE /v1/finetune/jobs/{job_id}
```

### Validate Dataset
```
POST /v1/finetune/validate
```

### List Chat Templates
```
GET /v1/finetune/templates
```

### Get Model Template Recommendation
```
GET /v1/finetune/templates/{model_name}
```

## Dataset Formats

The addon supports multiple dataset formats with automatic detection:

### ShareGPT
```json
{
  "conversations": [
    {"from": "human", "value": "Question"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

### Chat (OpenAI-style)
```json
{
  "messages": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ]
}
```

### Alpaca
```json
{
  "instruction": "Translate to French",
  "input": "Hello",
  "output": "Bonjour"
}
```

### Raw Text (for CPT)
```json
{
  "text": "Raw training text..."
}
```

## Output Structure

Training outputs are saved to `~/.llamafarm/models/llm/{job_id}/`:

```
~/.llamafarm/models/llm/{job_id}/
  config.json          # Job configuration
  job_config.json      # Detailed training config
  checkpoints/         # Training checkpoints
  lora_adapter/        # LoRA weights
  gguf/                # GGUF export (if enabled)
  training.log         # Training logs
  training_log.jsonl   # Loss per step
  result.json          # Final result
```

## Important Limitations

### GGUF Input Not Supported

**Unsloth requires HuggingFace format models.** GGUF models cannot be directly fine-tuned.

❌ **DON'T:**
```json
{
  "model": "Qwen/Qwen3-8B-GGUF"
}
```

✅ **DO:**
```json
{
  "model": "Qwen/Qwen3-8B"
}
```

The addon will detect GGUF model names and suggest the base model. After training, you can export to GGUF.

## Configuration

### LoRA Parameters

- `lora_rank`: LoRA rank (default: 16, range: 1-256)
- `lora_alpha`: LoRA alpha (default: 16)
- `target_modules`: Which layers to fine-tune (auto-detected if not specified)

For **SFT**: Targets attention and MLP layers  
For **CPT**: Also targets embeddings and lm_head

### Training Parameters

- `epochs`: Number of training epochs (default: 3 for SFT, 1 for CPT)
- `batch_size`: Batch size (default: 2)
- `learning_rate`: Learning rate (default: 2e-4 for SFT, 5e-5 for CPT)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `warmup_steps`: Warmup steps (default: 10)
- `gradient_accumulation_steps`: Gradient accumulation (default: 2)

### Dataset Limits

- Max examples: 100,000
- Max inline dataset size: ~50MB (for larger datasets, use file paths)

## Technical Details

### Backend Detection

```python
from finetune.trainer import detect_backend

backend = detect_backend()  # "mlx" on macOS, "cuda" otherwise
```

### Job Queue

Jobs are processed **sequentially** (one at a time) to prevent resource conflicts. This follows the same pattern as the vision training addon.

### OOM Isolation

Training runs in a subprocess to isolate out-of-memory errors:

1. Main process queues the job
2. Worker spawns subprocess with training script
3. If OOM occurs, subprocess crashes (not the runtime)
4. Main process reads result and continues

## Example Usage

### Python

```python
import requests

# Create SFT job
response = requests.post("http://localhost:11540/v1/finetune/sft", json={
    "model": "meta-llama/Llama-3-8B",
    "dataset": [
        {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "AI stands for Artificial Intelligence..."}
            ]
        }
    ],
    "epochs": 3,
    "output_gguf": True
})

job = response.json()
job_id = job["job_id"]

# Check status
status = requests.get(f"http://localhost:11540/v1/finetune/jobs/{job_id}").json()
print(f"Status: {status['status']}")
print(f"Progress: {status['progress'] * 100}%")
```

### cURL

```bash
# Create SFT job
curl -X POST http://localhost:11540/v1/finetune/sft \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3-8B",
    "dataset": [{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}],
    "epochs": 3
  }'

# List jobs
curl http://localhost:11540/v1/finetune/jobs

# Get job status
curl http://localhost:11540/v1/finetune/jobs/{job_id}
```

## Troubleshooting

### "Fine-tuning addon unavailable"

Install unsloth:
- macOS: `pip install unsloth-mlx`
- Linux/Windows: `pip install unsloth`

### "GGUF input not supported"

Remove `-GGUF` from model name. Use the base HuggingFace model.

### OOM errors

Reduce:
- `batch_size` (try 1)
- `max_seq_length` (try 1024 or 512)
- Dataset size

### Slow training

Normal on CPU. For faster training:
- Use GPU (NVIDIA or Apple Silicon)
- Reduce `max_seq_length`
- Reduce dataset size

## Testing

Run tests:
```bash
cd runtimes/universal
uv run python -m pytest tests/test_finetune.py -v
```

Tests cover:
- Dataset format detection and validation
- Format conversion (ShareGPT, Alpaca, Chat, Raw)
- Template compatibility checking
- Training time estimation
- Model validation
- Job queue lifecycle

## Architecture

```
finetune/
  __init__.py
  data_prep.py       # Dataset validation and formatting
  helpers.py         # Template detection, model validation
  trainer.py         # Job queue and trainer orchestration
  training_worker.py # Subprocess training script

routers/finetune/
  __init__.py
  types.py           # Pydantic models
  router.py          # FastAPI endpoints
```

## License

Part of LlamaFarm Universal Runtime.
