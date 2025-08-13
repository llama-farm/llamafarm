# LlamaFactory Fine-Tuner

The LlamaFactory Fine-Tuner provides streamlined integration with the LlamaFactory framework for efficient model fine-tuning.

## Features

- **Optimized Training Pipeline**: Pre-configured for optimal performance
- **Multiple Fine-tuning Methods**: LoRA, QLoRA, and full fine-tuning
- **Efficient Memory Usage**: Advanced quantization and optimization techniques
- **Easy Configuration**: Simplified YAML-based configuration
- **Multi-Model Support**: Works with LLaMA, Mistral, CodeLlama, and more

## Installation

```bash
pip install llamafactory-cli
```

## Configuration

### Basic LoRA Configuration

```yaml
type: "llamafactory"
config:
  base_model:
    name: "llama3.2-3b"
    
  method:
    type: "lora"
    r: 16
    alpha: 32
    dropout: 0.1
    target_modules: ["q_proj", "v_proj"]
    
  dataset:
    path: "./data/training.jsonl"
    conversation_template: "alpaca"
    
  training_args:
    output_dir: "./outputs/llama-lora"
    num_train_epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    learning_rate: 2e-4
    logging_steps: 10
    save_steps: 500
    bf16: true
    max_seq_length: 1024
```

### QLoRA Configuration (Memory Efficient)

```yaml
type: "llamafactory"
config:
  base_model:
    name: "llama3.1-8b"
    
  method:
    type: "qlora"
    r: 64
    alpha: 128
    dropout: 0.1
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    
  training_args:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 8
    bf16: true
    max_seq_length: 2048
```

## Supported Models

LlamaFactory supports a wide range of models:

- **LLaMA Family**: LLaMA 3.2 (3B), LLaMA 3.1 (8B, 70B)
- **Mistral**: Mistral-7B-Instruct
- **CodeLlama**: CodeLlama-13B-Instruct
- **Chinese Models**: Baichuan, ChatGLM, Qwen

## Usage Examples

### Basic Usage

```python
from models.components.fine_tuners.llamafactory import LlamaFactoryFineTuner

# Initialize with configuration
config = {
    "base_model": {
        "name": "llama3.2-3b"
    },
    "method": {
        "type": "lora",
        "r": 16,
        "alpha": 32
    },
    "dataset": {
        "path": "./data/training.jsonl",
        "conversation_template": "alpaca"
    },
    "training_args": {
        "output_dir": "./outputs",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4
    }
}

trainer = LlamaFactoryFineTuner(config)

# Start training
job = trainer.start_training()
print(f"Training started: {job.job_id}")

# Check status
status = trainer.get_training_status()
print(f"Current epoch: {status.current_epoch}/{status.total_epochs}")
```

### Resource Estimation

```python
# Get resource estimates before training
estimates = trainer.estimate_resources()
print(f"Estimated GPU memory: {estimates['memory_gb']} GB")
print(f"Estimated training time: {estimates['training_time_hours']} hours")
print(f"Recommended GPU: {estimates['recommended_gpu']}")
```

## Dataset Formats

### Alpaca Format (Default)
```json
{
  "instruction": "Explain quantum computing",
  "input": "",
  "output": "Quantum computing is a type of computation that..."
}
```

### ShareGPT Format
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "assistant", "value": "Machine learning is..."}
  ]
}
```

## Performance Tips

1. **Use QLoRA for Large Models**: Reduces memory usage by 75%
2. **Enable Gradient Accumulation**: Simulate larger batch sizes
3. **Use BF16 Precision**: Better stability than FP16
4. **Optimize Sequence Length**: Use only what you need

## Advanced Features

### Custom Templates

LlamaFactory supports custom conversation templates:

```yaml
dataset:
  conversation_template: "custom"
  custom_template:
    system: "You are a helpful assistant."
    user: "Human: {input}"
    assistant: "Assistant: {output}"
```

### Multi-GPU Training

LlamaFactory automatically handles multi-GPU setups:

```yaml
environment:
  num_gpus: 4
  deepspeed_config: "./configs/deepspeed.json"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Switch from LoRA to QLoRA
   - Reduce batch size
   - Reduce sequence length
   - Enable gradient checkpointing

2. **Slow Training**
   - Check GPU utilization
   - Increase batch size if memory allows
   - Use gradient accumulation
   - Enable mixed precision (bf16)

3. **Model Not Converging**
   - Adjust learning rate (try 1e-4 to 5e-4)
   - Increase LoRA rank
   - Check dataset quality
   - Verify template matching

## Comparison with PyTorch Fine-Tuner

| Feature | LlamaFactory | PyTorch |
|---------|--------------|---------|
| Setup Complexity | Low | Medium |
| Memory Efficiency | High | Medium |
| Training Speed | Fast | Medium |
| Customization | Medium | High |
| Model Support | Wide | Wide |

## Dependencies

LlamaFactory will be installed with:
- transformers
- peft
- datasets
- accelerate
- bitsandbytes (for QLoRA)
- deepspeed (optional)