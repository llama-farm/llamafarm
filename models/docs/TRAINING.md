# Training Guide - Deep Dive

This guide covers all aspects of model training in LlamaFarm, with a focus on PyTorch fine-tuning and configuration options.

## Table of Contents
- [Overview](#overview)
- [Training Methods](#training-methods)
- [PyTorch Fine-Tuning Deep Dive](#pytorch-fine-tuning-deep-dive)
- [LlamaFactory Integration](#llamafactory-integration)
- [Data Preparation](#data-preparation)
- [Monitoring & Evaluation](#monitoring--evaluation)

## Overview

LlamaFarm supports two primary training methods:
1. **PyTorch Fine-Tuning** - Direct PyTorch/Transformers integration with LoRA/QLoRA
2. **LlamaFactory** - Advanced fine-tuning framework with multiple methods

## Training Methods

### Quick Start

```bash
# Using a training strategy
uv run python cli.py train \
  --strategy demo3_training \
  --dataset your_data.jsonl \
  --epochs 3

# Using PyTorch directly
uv run python cli.py finetune start \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset train.jsonl \
  --method lora
```

## PyTorch Fine-Tuning Deep Dive

### Configuration Parameters

#### Model Configuration

```yaml
components:
  fine_tuner:
    type: pytorch
    config:
      base_model:
        huggingface_id: "meta-llama/Llama-2-7b-hf"  # Model identifier
        cache_dir: "./model_cache"                    # Where to cache models
        torch_dtype: "float16"                        # Precision: float16, bfloat16, float32
        load_in_8bit: false                          # 8-bit quantization
        load_in_4bit: false                          # 4-bit quantization (QLoRA)
        device_map: "auto"                           # Device placement strategy
        trust_remote_code: false                     # Allow custom model code
        use_auth_token: "${HF_TOKEN}"               # HuggingFace authentication
```

#### Training Method Configuration

##### LoRA (Low-Rank Adaptation)

```yaml
method:
  type: "lora"
  
  # LoRA Hyperparameters
  r: 16                          # Rank (higher = more parameters)
  alpha: 32                      # Scaling factor (alpha/r = actual scaling)
  dropout: 0.1                   # LoRA dropout for regularization
  
  # Target Modules (which layers to adapt)
  target_modules:                # If not specified, uses model defaults
    - "q_proj"                   # Query projection
    - "v_proj"                   # Value projection
    - "k_proj"                   # Key projection
    - "o_proj"                   # Output projection
    - "gate_proj"                # MLP gate (for Llama)
    - "up_proj"                  # MLP up projection
    - "down_proj"                # MLP down projection
  
  # Advanced LoRA Settings
  bias: "none"                   # Bias training: none, all, lora_only
  lora_alpha: 32                 # Same as alpha
  lora_dropout: 0.1              # Same as dropout
  modules_to_save: []            # Additional modules to train fully
  layers_to_transform: null      # Specific layer indices
  layers_pattern: null           # Regex pattern for layers
```

##### QLoRA (Quantized LoRA)

```yaml
method:
  type: "qlora"
  
  # QLoRA = LoRA + 4-bit quantization
  r: 16
  alpha: 32
  dropout: 0.1
  
  # Quantization Configuration
  quantization:
    load_in_4bit: true           # Enable 4-bit loading
    bnb_4bit_compute_dtype: "float16"  # Computation dtype
    bnb_4bit_quant_type: "nf4"  # Quantization type: fp4 or nf4
    bnb_4bit_use_double_quant: true  # Double quantization
```

#### Training Hyperparameters

```yaml
training:
  # Batch Size and Accumulation
  batch_size: 4                   # Per-device batch size
  gradient_accumulation_steps: 4  # Effective batch = batch_size * this
  gradient_checkpointing: true    # Memory optimization
  
  # Learning Rate
  learning_rate: 2e-4             # Initial learning rate
  lr_scheduler_type: "cosine"     # Scheduler: linear, cosine, polynomial
  warmup_ratio: 0.1               # Warmup as fraction of total steps
  warmup_steps: 0                 # Or fixed warmup steps (overrides ratio)
  
  # Training Duration
  num_epochs: 3                   # Number of epochs
  max_steps: -1                   # Or fixed steps (overrides epochs)
  
  # Optimization
  optimizer: "adamw_torch"        # Optimizer type
  adam_beta1: 0.9                # Adam beta1
  adam_beta2: 0.999              # Adam beta2
  adam_epsilon: 1e-8             # Adam epsilon
  weight_decay: 0.01             # L2 regularization
  max_grad_norm: 1.0             # Gradient clipping
  
  # Evaluation
  eval_strategy: "steps"         # When to evaluate: steps, epoch, no
  eval_steps: 100                # Evaluate every N steps
  eval_accumulation_steps: 4    # Gradient accumulation for eval
  per_device_eval_batch_size: 4 # Eval batch size
  
  # Checkpointing
  save_strategy: "steps"         # When to save: steps, epoch, no
  save_steps: 500                # Save every N steps
  save_total_limit: 3            # Keep only N checkpoints
  load_best_model_at_end: true  # Load best checkpoint when done
  metric_for_best_model: "loss" # Metric to determine best model
  greater_is_better: false       # Direction of metric improvement
  
  # Logging
  logging_steps: 10              # Log every N steps
  logging_first_step: true       # Log the first step
  report_to: ["tensorboard"]     # Where to log: tensorboard, wandb, none
  
  # Performance
  fp16: true                     # Mixed precision training
  bf16: false                    # BFloat16 (requires Ampere GPUs)
  tf32: true                     # TensorFloat-32 (Ampere GPUs)
  dataloader_num_workers: 4     # Parallel data loading
  dataloader_pin_memory: true   # Pin memory for faster GPU transfer
  
  # Memory Optimization
  gradient_checkpointing: true   # Trade compute for memory
  optim: "adamw_8bit"           # 8-bit optimizer (saves memory)
  group_by_length: true          # Group similar-length sequences
  length_column_name: "length"  # Column with sequence lengths
  
  # Advanced Options
  seed: 42                       # Random seed for reproducibility
  data_seed: null               # Separate seed for data sampling
  push_to_hub: false            # Push to HuggingFace Hub
  hub_model_id: null            # Hub model name
  hub_strategy: "every_save"    # When to push: every_save, checkpoint, end
  hub_token: "${HF_TOKEN}"      # Hub authentication
```

### Hardware-Specific Optimizations

#### NVIDIA GPUs (CUDA)

```yaml
optimization:
  cuda:
    use_flash_attention: true    # Flash Attention 2 (Ampere+)
    use_bf16: true               # BFloat16 on Ampere+
    gradient_checkpointing: true
    optim: "adamw_8bit"
    per_device_batch_size: 8    # Can be higher with optimization
```

#### Apple Silicon (MPS)

```yaml
optimization:
  mps:
    use_cpu_offload: true        # Offload to CPU when needed
    torch_dtype: "float32"       # MPS works best with float32
    per_device_batch_size: 2
    gradient_accumulation_steps: 8
```

#### CPU-Only

```yaml
optimization:
  cpu:
    torch_dtype: "float32"
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    use_ipex: true               # Intel Extension for PyTorch
```

### Memory Optimization Techniques

1. **Gradient Checkpointing**: Recompute activations during backward pass
2. **8-bit Optimizers**: Reduce optimizer memory by 75%
3. **QLoRA**: 4-bit quantization reduces memory by ~75%
4. **CPU Offloading**: Move optimizer states to CPU
5. **Gradient Accumulation**: Simulate larger batches
6. **Mixed Precision**: Use FP16/BF16 for 50% memory savings

### Example Configurations

#### Minimal Memory (Consumer GPU)

```yaml
fine_tuner:
  type: pytorch
  config:
    base_model:
      huggingface_id: "meta-llama/Llama-2-7b-hf"
      load_in_4bit: true
    method:
      type: qlora
      r: 8
      alpha: 16
    training:
      batch_size: 1
      gradient_accumulation_steps: 16
      gradient_checkpointing: true
      optim: "paged_adamw_8bit"
      fp16: true
```

#### Maximum Performance (A100/H100)

```yaml
fine_tuner:
  type: pytorch
  config:
    base_model:
      huggingface_id: "meta-llama/Llama-2-70b-hf"
      torch_dtype: "bfloat16"
    method:
      type: lora
      r: 64
      alpha: 128
    training:
      batch_size: 16
      gradient_accumulation_steps: 1
      bf16: true
      tf32: true
      use_flash_attention: true
```

## LlamaFactory Integration

LlamaFactory provides additional fine-tuning methods:

```yaml
fine_tuner:
  type: llamafactory
  config:
    model_name_or_path: "meta-llama/Llama-2-7b-hf"
    finetuning_type: "lora"     # full, freeze, lora, qlora
    template: "llama2"           # Model-specific template
    dataset: "alpaca"            # Or path to custom dataset
    
    # LlamaFactory-specific options
    quantization_bit: 4          # 4 or 8 bit quantization
    double_quantization: true    
    lora_rank: 8
    lora_alpha: 16
    lora_dropout: 0.1
    lora_target: "q_proj,v_proj"
    
    # Advanced training options
    stage: "sft"                 # sft, rm, ppo, dpo
    packing: false               # Pack multiple examples
    upcast_layernorm: true      # Upcast layernorm for stability
    use_unsloth: false          # Use Unsloth optimization
```

## Data Preparation

### Dataset Format

Training data should be in JSONL format:

```json
{"instruction": "What is machine learning?", "output": "Machine learning is..."}
{"instruction": "Explain neural networks", "output": "Neural networks are..."}
```

### Data Splitting

```bash
# Create train/eval split (90/10)
uv run python cli.py datasplit \
  --input data.jsonl \
  --train-output train.jsonl \
  --eval-output eval.jsonl \
  --eval-percentage 10
```

### Data Quality Checks

1. **Minimum examples**: At least 50-100 for LoRA
2. **Balanced lengths**: Similar sequence lengths train better
3. **Clean text**: Remove special characters, fix encoding
4. **Diverse examples**: Avoid repetition

## Monitoring & Evaluation

### Training Metrics

- **Loss**: Should decrease over time
- **Learning Rate**: Monitor scheduler behavior
- **Gradient Norm**: Check for exploding gradients
- **GPU Memory**: Optimize if OOM errors

### Using TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./output/runs

# Training logs to this directory automatically
```

### Evaluation Metrics

```yaml
training:
  # Evaluation configuration
  eval_strategy: "steps"
  eval_steps: 100
  
  # Metrics to compute
  compute_metrics: true
  metric_for_best_model: "eval_loss"
  
  # Early stopping
  early_stopping: true
  early_stopping_patience: 3
  early_stopping_threshold: 0.0001
```

### Model Testing

```bash
# Test the fine-tuned model
uv run python cli.py generate \
  --model ./output/checkpoint-best \
  --prompt "Your test prompt"
```

## Troubleshooting

### Common Issues

1. **OOM (Out of Memory)**
   - Reduce batch_size
   - Enable gradient_checkpointing
   - Use QLoRA instead of LoRA
   - Reduce sequence length

2. **Slow Training**
   - Enable mixed precision (fp16/bf16)
   - Use Flash Attention
   - Increase dataloader_num_workers
   - Check GPU utilization

3. **Poor Results**
   - Increase training epochs
   - Adjust learning rate
   - Use larger LoRA rank
   - Check data quality

4. **Gradient Explosion**
   - Reduce learning rate
   - Enable gradient clipping
   - Use warmup steps

## Best Practices

1. **Start Small**: Test with subset of data first
2. **Monitor Actively**: Watch loss curves and metrics
3. **Save Checkpoints**: Keep best and last checkpoints
4. **Validate Often**: Regular evaluation prevents overfitting
5. **Document Experiments**: Track hyperparameters and results