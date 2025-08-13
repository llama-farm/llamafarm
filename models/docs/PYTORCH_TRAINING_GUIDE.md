# 🔥 PyTorch Fine-Tuning Complete Guide for LlamaFarm

## Table of Contents
- [Overview](#overview)
- [LlamaFarm Strategy Integration](#llamafarm-strategy-integration)
- [Hardware Requirements](#hardware-requirements)
- [Training Parameters](#training-parameters)
- [Hardware-Specific Settings](#hardware-specific-settings)
- [Use-Case Configurations](#use-case-configurations)
- [Evaluation & Metrics](#evaluation--metrics)
- [Strategy Examples](#strategy-examples)
- [CLI Commands](#cli-commands)
- [Troubleshooting](#troubleshooting)

## Overview

This guide provides comprehensive documentation for PyTorch-based fine-tuning in LlamaFarm's strategy system. All configurations are defined in strategy YAML files and executed through the LlamaFarm CLI.

## LlamaFarm Strategy Integration

### How Strategies Work

In LlamaFarm, all training configurations are defined as **strategies** in YAML files. This provides:
- **Reusability**: Define once, use anywhere
- **Version Control**: Track configuration changes
- **Modularity**: Mix and match components
- **Simplicity**: No code changes needed

### Strategy Structure for PyTorch Training

```yaml
strategy_name:
  name: "Human-readable name"
  description: "What this strategy does"
  
  components:
    fine_tuner:
      type: pytorch  # Selects PyTorch fine-tuner
      config:
        hardware:     # Hardware-specific settings
        method:       # LoRA configuration
        training_args: # Training hyperparameters
        
    model_app:        # Optional: deployment component
      type: ollama
      config:
        # Deployment settings
        
  dataset:            # Data configuration
    train_file: path/to/train.jsonl
    eval_file: path/to/eval.jsonl
    eval_split: 0.1
    
  export:             # Model export settings
    to_ollama: true
    to_gguf: true
    
  prompt_template:    # Prompt formatting
    format: "..."
```

### Using Strategies via CLI

```bash
# Train with a strategy
uv run python cli.py train --strategy your_strategy_name --strategy-file strategies.yaml

# Test a strategy configuration
uv run python cli.py setup strategies.yaml --strategy your_strategy_name --verify-only

# Complete training pipeline
uv run python cli.py complete "test prompt" --strategy your_strategy_name --strategy-file strategies.yaml
```

## Hardware Requirements

### Minimum Requirements by Model Size

| Model Size | VRAM/RAM | Storage | Training Time* |
|------------|----------|---------|----------------|
| 1B params  | 4GB      | 10GB    | 30-60 min      |
| 3B params  | 8GB      | 20GB    | 1-2 hours      |
| 7B params  | 16GB     | 40GB    | 2-4 hours      |
| 13B params | 24GB     | 80GB    | 4-8 hours      |

*Training time for 1000 examples with LoRA on consumer hardware

### Supported Hardware Platforms

| Platform | Device Type | Optimal For | Key Features |
|----------|------------|-------------|--------------|
| NVIDIA GPU | CUDA | All model sizes | FP16/BF16, Fast training |
| Apple Silicon | MPS | ≤7B models | Unified memory, Energy efficient |
| AMD GPU | ROCm | Medium models | Good price/performance |
| CPU Only | CPU | Small models/testing | Universal compatibility |

## Training Parameters

### Core LoRA Parameters

| Parameter | Range | Default | Description | When to Adjust |
|-----------|-------|---------|-------------|----------------|
| `r` (rank) | 4-64 | 16 | LoRA matrix rank | Lower for small models/datasets |
| `alpha` | r to 2*r | 32 | LoRA scaling factor | Usually 2*r for stability |
| `dropout` | 0.0-0.2 | 0.1 | Regularization dropout | Higher to prevent overfitting |
| `target_modules` | List | [q,v,k,o] | Modules to apply LoRA | Add gate/up/down for better learning |

#### Recommended LoRA Settings by Model Size

| Model Size | Rank (r) | Alpha | Dropout | Target Modules |
|------------|----------|-------|---------|----------------|
| <1B | 4 | 8 | 0.1 | [q_proj, v_proj] |
| 1-3B | 8 | 16 | 0.05 | [q_proj, v_proj, k_proj, o_proj] |
| 3-7B | 16 | 32 | 0.05 | [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj] |
| 7-13B | 32 | 64 | 0.1 | All linear layers |

### Training Hyperparameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| `learning_rate` | Step size for optimization | 1e-5 to 5e-4 | **Critical** - Too high causes instability |
| `num_train_epochs` | Training iterations over dataset | 1-5 | More epochs risk overfitting |
| `per_device_train_batch_size` | Examples per GPU/iteration | 1-8 | Limited by memory |
| `gradient_accumulation_steps` | Accumulate gradients | 1-16 | Simulate larger batches |
| `warmup_ratio` | Fraction of steps for LR warmup | 0.03-0.1 | Helps stability |
| `weight_decay` | L2 regularization | 0.0-0.01 | Prevents overfitting |

### Evaluation Parameters

| Parameter | Description | Recommended | Purpose |
|-----------|-------------|-------------|---------|
| `eval_strategy` | When to evaluate | "steps" | Regular evaluation |
| `eval_steps` | Evaluation frequency | 50-100 | Balance between overhead and monitoring |
| `eval_accumulation_steps` | Gradient accumulation for eval | Same as training | Consistency |
| `metric_for_best_model` | Metric to select best checkpoint | "eval_loss" | Model selection |
| `load_best_model_at_end` | Load best checkpoint | true | Use best, not last |

### Generation Control Parameters

| Parameter | Description | Range | Effect |
|-----------|-------------|-------|--------|
| `max_length` | Maximum sequence length | 128-512 | Prevents runaway generation |
| `max_new_tokens` | Max tokens to generate | 50-200 | Direct output control |
| `temperature` | Randomness in generation | 0.1-1.0 | Lower = more deterministic |
| `top_p` | Nucleus sampling | 0.9-0.95 | Token selection diversity |
| `repetition_penalty` | Penalize repeated tokens | 1.0-1.2 | Reduce repetition |

## Hardware-Specific Settings

### NVIDIA GPU (CUDA) Configuration

```yaml
hardware:
  device: cuda
  precision:
    cuda:
      use_fp16: true      # Fast, memory efficient
      use_bf16: false     # Only for Ampere+ GPUs
      use_tf32: true      # Tensor Core acceleration
  optimization:
    gradient_checkpointing: true  # If <12GB VRAM
    mixed_precision: fp16
```

#### CUDA GPU Recommendations

| GPU Model | VRAM | Batch Size | FP16 | BF16 | Max Model |
|-----------|------|------------|------|------|-----------|
| RTX 3060 | 12GB | 2-4 | ✅ | ❌ | 7B |
| RTX 3070 | 8GB | 1-2 | ✅ | ❌ | 3B |
| RTX 3080 | 10GB | 2-4 | ✅ | ✅ | 7B |
| RTX 3090 | 24GB | 4-8 | ✅ | ✅ | 13B |
| RTX 4070 Ti | 12GB | 2-4 | ✅ | ✅ | 7B |
| RTX 4080 | 16GB | 4-6 | ✅ | ✅ | 7B |
| RTX 4090 | 24GB | 6-8 | ✅ | ✅ | 13B |
| A100 | 40-80GB | 8-16 | ✅ | ✅ | 30B+ |

### Apple Silicon (MPS) Configuration

```yaml
hardware:
  device: mps
  precision:
    mps:
      use_fp16: false     # Not stable on MPS
      use_bf16: false     # Not supported
  optimization:
    gradient_checkpointing: false  # Usually not needed
    mixed_precision: no
```

#### Apple Silicon Recommendations

| Chip | Memory | Batch Size | Recommended Models | Notes |
|------|--------|------------|-------------------|--------|
| M1 | 8GB | 1 | ≤1B | Basic fine-tuning |
| M1 | 16GB | 1-2 | ≤3B | Good for small models |
| M1 Pro/Max | 32GB | 2-4 | ≤7B | Excellent performance |
| M2 | 8-24GB | 1-2 | 1-3B | 20% faster than M1 |
| M2 Pro/Max | 32GB | 2-4 | ≤7B | Best for development |
| M3 | 8-24GB | 1-2 | 1-3B | Latest generation |
| M3 Pro/Max | 36-128GB | 4-8 | 7-13B | Production capable |

### CPU-Only Configuration

```yaml
hardware:
  device: cpu
  precision:
    cpu:
      use_fp16: false     # Keep full precision
  optimization:
    gradient_checkpointing: true
    gradient_accumulation_steps: 8  # Simulate larger batches
    mixed_precision: no
```

## Use-Case Configurations

### 1. Quick Experimentation

**Goal**: Fast iteration, testing ideas

```yaml
method:
  type: lora
  r: 4
  alpha: 8
  dropout: 0.05

training_args:
  num_train_epochs: 1
  per_device_train_batch_size: 2
  learning_rate: 0.0002
  eval_strategy: "no"  # Skip evaluation for speed
  save_strategy: "no"  # Don't save checkpoints
```

### 2. Production Fine-Tuning

**Goal**: Best quality, robust model

```yaml
method:
  type: lora
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

training_args:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.00005
  warmup_ratio: 0.1
  weight_decay: 0.01
  eval_strategy: "steps"
  eval_steps: 50
  save_strategy: "steps"
  save_steps: 100
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
```

### 3. Small Dataset (<100 examples)

**Goal**: Prevent overfitting, maximize learning

```yaml
method:
  type: lora
  r: 4  # Very low rank
  alpha: 8
  dropout: 0.15  # Higher dropout

training_args:
  num_train_epochs: 5-10  # More epochs OK with small data
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 0.00001  # Very low LR
  weight_decay: 0.02  # Strong regularization
  eval_strategy: "epoch"
  load_best_model_at_end: true
```

### 4. Large Dataset (>10,000 examples)

**Goal**: Efficient training, good generalization

```yaml
method:
  type: lora
  r: 32  # Higher rank for complex patterns
  alpha: 64
  dropout: 0.05  # Lower dropout (data provides regularization)

training_args:
  num_train_epochs: 1-2  # Fewer epochs needed
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 0.0002
  eval_strategy: "steps"
  eval_steps: 500  # Less frequent evaluation
  save_steps: 1000
```

### 5. Domain Adaptation (Medical, Legal, etc.)

**Goal**: Specialized knowledge while preserving general capabilities

```yaml
method:
  type: lora
  r: 8
  alpha: 16
  dropout: 0.1
  target_modules: [q_proj, v_proj, k_proj, o_proj]  # Core attention only

training_args:
  num_train_epochs: 2-3
  per_device_train_batch_size: 2
  learning_rate: 0.00005
  warmup_ratio: 0.05
  eval_split: 0.15  # Larger eval for domain validation
  
data_preprocessing:
  max_length: 256  # Shorter for focused responses
  remove_duplicates: true
  filter_quality: true  # Remove low-quality examples
```

### 6. Conversational AI

**Goal**: Natural dialogue while preventing multi-turn generation

```yaml
method:
  type: lora
  r: 12
  alpha: 24
  dropout: 0.08

training_args:
  num_train_epochs: 2
  learning_rate: 0.0001
  max_new_tokens: 150  # Prevent long responses
  temperature: 0.7  # Natural variation
  repetition_penalty: 1.1
  
prompt_template:
  format: |
    User: {input}
    Assistant: {output}
  stop_sequences: ["\nUser:", "\nAssistant:", "###"]
```

## Evaluation & Metrics

### Train/Eval Split Strategies

| Split Ratio | Use Case | Pros | Cons |
|------------|----------|------|------|
| 95/5 | Very small datasets | Maximum training data | Less robust evaluation |
| 90/10 | Standard (recommended) | Balanced | Good for most cases |
| 85/15 | Important validation | More robust metrics | Less training data |
| 80/20 | Research/publishing | Very robust | Significantly less training |
| 70/30 | Hyperparameter tuning | Reliable for tuning | Much less training data |

### Creating Data Splits

```bash
# Standard 90/10 split
python demos/create_data_split.py --input data.jsonl --eval-percent 10

# Robust 85/15 split for production
python demos/create_data_split.py --input data.jsonl --eval-percent 15

# Small dataset with 5% eval
python demos/create_data_split.py --input data.jsonl --eval-percent 5
```

### Evaluation Metrics to Monitor

| Metric | What It Measures | Good Values | Warning Signs |
|--------|-----------------|-------------|---------------|
| `train_loss` | Training fit | Decreasing | Plateau or increase |
| `eval_loss` | Generalization | Close to train_loss | Much higher than train |
| `eval_perplexity` | Prediction confidence | <50 for good models | >100 indicates problems |
| `learning_rate` | Current LR | Per schedule | Unexpected jumps |

### Overfitting Indicators

1. **Gap Analysis**: `eval_loss - train_loss > 0.5`
2. **Evaluation Degradation**: Eval loss increases while train decreases
3. **Perfect Training**: Train loss near 0 but poor eval
4. **Repetitive Output**: Model generates same phrases repeatedly

## Memory Optimization Techniques

### Gradient Checkpointing

Trades computation for memory by recomputing activations during backward pass.

```yaml
optimization:
  gradient_checkpointing: true  # Enable for large models/small memory
```

**When to use:**
- GPU has <12GB VRAM
- Training 7B+ models
- Batch size limited by memory

**Trade-off:** ~20% slower training for 30-50% memory savings

### Gradient Accumulation

Simulates larger batch sizes without requiring more memory.

```yaml
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch size = 8
```

**Effective Batch Size** = `batch_size * gradient_accumulation_steps * num_devices`

### Mixed Precision Training

Uses 16-bit floats for computation, 32-bit for model weights.

```yaml
optimization:
  mixed_precision: fp16  # or bf16 for newer GPUs
```

**Benefits:**
- 50% memory reduction
- 2-3x faster on modern GPUs
- Minimal accuracy loss

## Common Issues & Solutions

### Out of Memory (OOM) Errors

| Solution | Memory Savings | Performance Impact |
|----------|---------------|-------------------|
| Reduce batch size | High | None |
| Enable gradient checkpointing | 30-50% | 20% slower |
| Reduce sequence length | Quadratic | Limits context |
| Use gradient accumulation | None | None |
| Lower LoRA rank | Moderate | May reduce quality |
| Enable mixed precision | 50% | 2-3x faster |

### Slow Training

| Issue | Solution | Impact |
|-------|----------|--------|
| CPU bottleneck | Pre-process data, use DataLoader workers | 2-5x speedup |
| Small batch size | Increase gradient accumulation | No memory increase |
| Checkpointing overhead | Save less frequently | Fewer interruptions |
| Evaluation too frequent | Increase eval_steps | Less overhead |

### Poor Model Quality

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| High eval loss | Overfitting | Add dropout, reduce epochs |
| Repetitive output | High learning rate | Lower LR, add repetition penalty |
| Forgets base knowledge | LoRA rank too high | Reduce rank, fewer target modules |
| Doesn't learn task | LR too low or rank too low | Increase carefully |
| Unstable training | LR too high | Reduce LR, add warmup |

## Advanced Configurations

### Multi-GPU Training

```yaml
training_args:
  ddp_find_unused_parameters: false  # Faster DDP
  dataloader_num_workers: 4  # Parallel data loading
  fp16_backend: "auto"  # Automatic mixed precision backend
```

### Custom Scheduling

```yaml
training_args:
  lr_scheduler_type: "cosine"  # Better than linear for most cases
  warmup_ratio: 0.1
  num_train_epochs: 3
```

### Data Preprocessing

```yaml
data_preprocessing:
  max_input_length: 256
  max_output_length: 256
  remove_duplicates: true
  shuffle: true
  seed: 42  # Reproducible shuffling
```

## Best Practices Checklist

### Before Training
- [ ] Create train/eval split (90/10 recommended)
- [ ] Check GPU memory with `nvidia-smi` or Activity Monitor
- [ ] Verify base model loads successfully
- [ ] Review dataset for quality and format
- [ ] Set appropriate evaluation frequency

### During Training
- [ ] Monitor loss curves for overfitting
- [ ] Check GPU utilization stays >80%
- [ ] Watch for OOM errors early
- [ ] Verify checkpoints are being saved

### After Training
- [ ] Compare train vs eval metrics
- [ ] Test model on holdout examples
- [ ] Convert to deployment format (GGUF/Ollama)
- [ ] Document hyperparameters used
- [ ] Save best checkpoint separately

## Environment Variables

```bash
# Hugging Face
export HF_TOKEN="your_token"
export HF_HOME="./cache"

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # For OOM issues
export CUDA_VISIBLE_DEVICES="0"  # Select GPU

# Logging
export TRANSFORMERS_VERBOSITY="info"
export WANDB_MODE="offline"  # For offline training
```

## Strategy Examples

### Complete LlamaFarm Strategy Examples

#### 1. Medical Domain Adaptation (demo3_training_optimized)
```yaml
demo3_training_optimized:
  name: "Optimized Medical Model Training"
  description: "Prevents conversation generation with conservative settings"
  
  components:
    fine_tuner:
      type: pytorch
      config:
        hardware:
          device: auto
          memory_gb: 8
          optimization:
            gradient_checkpointing: false
            gradient_accumulation_steps: 2
            
        method:
          type: lora
          r: 4              # Very low rank for 3B model
          alpha: 8          # 2*r for stability
          dropout: 0.1      # Higher dropout for regularization
          target_modules: [q_proj, v_proj, k_proj, o_proj]
          
        training_args:
          output_dir: ./fine_tuned_models/medical_optimized
          num_train_epochs: 2
          per_device_train_batch_size: 1
          gradient_accumulation_steps: 4
          learning_rate: 0.00005
          warmup_steps: 10
          weight_decay: 0.01
          # Evaluation settings
          eval_strategy: "steps"
          eval_steps: 50
          load_best_model_at_end: true
          metric_for_best_model: "eval_loss"
          
  dataset:
    train_file: demos/datasets/medical/medical_qa_train.jsonl
    eval_file: demos/datasets/medical/medical_qa_eval.jsonl
    eval_split: 0.1
    
  export:
    to_ollama: true
    model_name: medical-llama3.2-optimized
    
  prompt_template:
    system: "Answer medical questions briefly and accurately."
    format: |
      Question: {input}
      Answer: {output}
```

#### 2. High-Performance GPU Training
```yaml
training_cuda_datacenter:
  name: "Datacenter GPU Training"
  description: "For A100/H100 or RTX 4090"
  
  components:
    fine_tuner:
      type: pytorch
      config:
        hardware:
          device: cuda
          memory_gb: 40
          precision:
            cuda:
              use_fp16: false
              use_bf16: true    # Better for large models
              use_tf32: true    # Tensor Core optimization
          optimization:
            gradient_checkpointing: false
            mixed_precision: bf16
            
        base_model:
          huggingface_id: meta-llama/Llama-2-7b-hf
          torch_dtype: bfloat16
          
        method:
          type: lora
          r: 32
          alpha: 64
          dropout: 0.05
          target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
          
        training_args:
          per_device_train_batch_size: 8
          gradient_accumulation_steps: 1
          num_train_epochs: 2
          learning_rate: 0.0002
          
  monitoring:
    track_training_metrics: true
    log_tensorboard: true
```

#### 3. Apple Silicon Optimized
```yaml
training_mps_apple:
  name: "Apple Silicon Training"
  description: "Optimized for M1/M2/M3 Macs"
  
  components:
    fine_tuner:
      type: pytorch
      config:
        hardware:
          device: mps
          memory_gb: 16
          precision:
            mps:
              use_fp16: false   # Not stable on MPS
              use_bf16: false   # Not supported
          optimization:
            gradient_checkpointing: false
            mixed_precision: no
            
        base_model:
          huggingface_id: meta-llama/Llama-3.2-3B-Instruct
          torch_dtype: float32  # Full precision for stability
          
        training_args:
          per_device_train_batch_size: 2
          gradient_accumulation_steps: 2
          learning_rate: 0.0001
```

### CLI Commands with Strategies

```bash
# 1. Setup and verify strategy
uv run python cli.py setup demos/strategies.yaml --strategy demo3_training_optimized --verify-only

# 2. Train with specific strategy
uv run python cli.py train \
  --strategy demo3_training_optimized \
  --strategy-file demos/strategies.yaml \
  --train-dataset medical_qa_train.jsonl \
  --eval-dataset medical_qa_eval.jsonl

# 3. Test trained model
uv run python cli.py complete "What are symptoms of diabetes?" \
  --strategy demo3_finetuned_model \
  --strategy-file demos/strategies.yaml

# 4. Convert to deployment format
uv run python cli.py convert \
  ./fine_tuned_models/medical_optimized/final_model \
  ./medical-model \
  --format ollama \
  --model-name medical-llama3.2

# 5. Run complete demo pipeline
python demos/demo3_training.py
```

### Strategy Selection Guide

| Use Case | Strategy | Key Features |
|----------|----------|--------------|
| Medical/Legal Domain | `demo3_training_optimized` | Low rank, high regularization, eval monitoring |
| High-End GPU | `training_cuda_datacenter` | BF16, large batches, tensor cores |
| Apple Silicon | `training_mps_apple` | FP32, conservative batches, MPS optimized |
| Consumer GPU | `training_cuda_consumer` | FP16, gradient checkpointing, memory efficient |
| CPU Only | `training_cpu_only` | Maximum memory optimization, small batches |
| Quick Testing | `demo_minimal` | 1 epoch, no eval, fast iteration |
| Production | `demo_production` | Full evaluation, best model selection, robust |

## References

- [Hugging Face Transformers Training](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

---

*For specific implementation details, see the [demo3_training.py](../demos/demo3_training.py) example.*