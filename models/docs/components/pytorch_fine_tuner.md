# PyTorch Fine-Tuner

The PyTorch Fine-Tuner provides a comprehensive implementation for fine-tuning language models using PyTorch and Hugging Face Transformers.

## Features

- **Multiple Fine-tuning Methods**:
  - Full fine-tuning
  - LoRA (Low-Rank Adaptation)
  - QLoRA (Quantized LoRA)

- **Model Support**:
  - LLaMA models
  - Mistral models
  - GPT-2 and variants
  - CodeLlama
  - Phi models
  - Qwen models

- **Advanced Training Features**:
  - Mixed precision training (FP16/BF16)
  - Gradient checkpointing
  - Multi-GPU support via device_map
  - Quantization support (4-bit, 8-bit)
  - Custom learning rate schedules
  - Gradient accumulation

## Configuration

### Basic Configuration

```yaml
type: "pytorch"
config:
  base_model:
    name: "llama-7b"
    huggingface_id: "meta-llama/Llama-2-7b-hf"
    torch_dtype: "bfloat16"
    device_map: "auto"
    
  method:
    type: "lora"  # or "qlora", "full_finetune"
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    
  dataset:
    path: "./data/training.jsonl"
    preprocessing_num_workers: 4
    
  training_args:
    output_dir: "./outputs/llama-lora"
    num_train_epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 2
    learning_rate: 2e-4
    warmup_ratio: 0.03
    logging_steps: 10
    save_steps: 500
    save_total_limit: 3
    fp16: false
    bf16: true
    max_seq_length: 512
```

### QLoRA Configuration

```yaml
type: "pytorch"
config:
  base_model:
    name: "llama-13b"
    huggingface_id: "meta-llama/Llama-2-13b-hf"
    torch_dtype: "bfloat16"
    device_map: "auto"
    
  method:
    type: "qlora"
    r: 64
    alpha: 128
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    
  framework:
    gradient_checkpointing: true
    use_fast_tokenizer: true
    
  training_args:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 4
    bf16: true
```

## Dataset Formats

### Alpaca Format
```json
{
  "instruction": "Write a function to calculate factorial",
  "input": "n = 5",
  "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
}
```

### Text Format
```json
{
  "text": "### Human: What is machine learning?\n### Assistant: Machine learning is..."
}
```

## Usage Examples

### Basic Usage

```python
from models.components.fine_tuners.pytorch import PyTorchFineTuner

# Initialize with configuration
config = {
    "base_model": {
        "name": "llama-7b",
        "huggingface_id": "meta-llama/Llama-2-7b-hf"
    },
    "method": {
        "type": "lora",
        "r": 16,
        "alpha": 32
    },
    "dataset": {
        "path": "./data/training.jsonl"
    },
    "training_args": {
        "output_dir": "./outputs",
        "num_train_epochs": 3
    }
}

trainer = PyTorchFineTuner(config)

# Validate configuration
errors = trainer.validate_config()
if errors:
    print("Configuration errors:", errors)
    exit(1)

# Start training
job = trainer.start_training()
print(f"Training started: {job.job_id}")

# Monitor progress
while job.status == "running":
    print(f"Epoch {job.current_epoch}/{job.total_epochs}, Step {job.current_step}/{job.total_steps}")
    time.sleep(10)

# Export model
trainer.export_model("./final_model")
```

### Resume Training

```python
# Resume from checkpoint
job = trainer.resume_training("./outputs/checkpoint-500")
```

## Performance Optimization

### Memory Optimization
- Use QLoRA for large models on limited GPU memory
- Enable gradient checkpointing
- Reduce batch size and increase gradient accumulation
- Use mixed precision training (bf16)

### Speed Optimization
- Use Flash Attention 2 (if supported)
- Enable TF32 on Ampere GPUs
- Optimize data loading with multiple workers
- Use cached tokenized datasets

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use QLoRA instead of LoRA
   - Reduce max sequence length

2. **Slow Training**
   - Check if using GPU properly
   - Enable mixed precision (bf16)
   - Increase number of data workers
   - Use SSD for dataset storage

3. **Model Not Learning**
   - Check learning rate (try 1e-4 to 5e-4)
   - Verify dataset format
   - Check if model is frozen
   - Monitor gradient norms

## Advanced Features

### Custom Training Loop

The PyTorch fine-tuner supports custom training callbacks:

```python
class CustomCallback(transformers.TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} completed!")
```

### Multi-GPU Training

Automatically handled with `device_map="auto"`. For specific GPU allocation:

```yaml
base_model:
  device_map: {
    "model.embed_tokens": 0,
    "model.layers.0-15": 0,
    "model.layers.16-31": 1,
    "lm_head": 1
  }
```

## Dependencies

- torch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.4.0
- datasets >= 2.12.0
- accelerate >= 0.20.0
- bitsandbytes >= 0.40.0 (for QLoRA)