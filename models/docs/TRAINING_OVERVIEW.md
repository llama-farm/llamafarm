# 🎓 LlamaFarm Training System Overview

## Quick Links
- [PyTorch Training Guide](./PYTORCH_TRAINING_GUIDE.md) - Comprehensive parameter reference
- [Strategy Examples](./STRATEGY_EXAMPLES.md) - Pre-built configurations
- [Demo Scripts](../demos/README.md) - Working examples
- [CLI Reference](./CLI.md) - Command documentation

## Training System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LlamaFarm CLI                      │
├─────────────────────────────────────────────────────┤
│                  Strategy System                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   PyTorch   │  │LlamaFactory │  │   Ollama    │ │
│  │ Fine-Tuner  │  │ Fine-Tuner  │  │  Converter  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────┤
│              Data Pipeline & Evaluation              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Dataset   │  │  Train/Eval │  │   Metrics   │ │
│  │   Loading   │  │    Split    │  │  Tracking   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. Strategy-Based Configuration
All training configurations are defined in YAML strategies:
- **Reusable**: Define once, use everywhere
- **Version Controlled**: Track changes over time
- **Modular**: Mix and match components
- **No Code Changes**: Pure configuration

### 2. Evaluation-Driven Training
Every training includes evaluation by default:
- **90/10 Split**: Standard train/eval division
- **Real-time Monitoring**: Track generalization
- **Best Model Selection**: Use optimal checkpoint
- **Overfitting Prevention**: Early detection

### 3. Hardware Optimization
Automatic detection and optimization for:
- **NVIDIA GPUs**: CUDA with FP16/BF16
- **Apple Silicon**: MPS with unified memory
- **CPU**: Optimized for available cores
- **Memory Management**: Gradient checkpointing, accumulation

## Quick Start Examples

### 1. Medical Domain Fine-Tuning
```bash
# Run the complete demo with evaluation
python demos/demo3_training.py

# Or use CLI directly
uv run python cli.py train \
  --strategy demo3_training_optimized \
  --strategy-file demos/strategies.yaml \
  --train-dataset medical_qa_train.jsonl \
  --eval-dataset medical_qa_eval.jsonl
```

### 2. Custom Training Configuration
```yaml
# my_strategy.yaml
my_training:
  name: "Custom Domain Training"
  components:
    fine_tuner:
      type: pytorch
      config:
        method:
          type: lora
          r: 8
          alpha: 16
        training_args:
          num_train_epochs: 3
          eval_strategy: "steps"
          eval_steps: 100
```

```bash
# Train with custom strategy
uv run python cli.py train --strategy my_training --strategy-file my_strategy.yaml
```

### 3. Data Preparation
```bash
# Create train/eval split
python demos/create_data_split.py --input data.jsonl --eval-percent 10

# Result: data_train.jsonl (90%) and data_eval.jsonl (10%)
```

## Training Workflow

### Step 1: Prepare Data
```bash
# Clean and format your dataset
python prepare_data.py --input raw_data.txt --output data.jsonl

# Create train/eval split
python demos/create_data_split.py --input data.jsonl --eval-percent 10
```

### Step 2: Configure Strategy
```yaml
# strategies.yaml
my_strategy:
  components:
    fine_tuner:
      type: pytorch
      config:
        # Your configuration
  dataset:
    train_file: data_train.jsonl
    eval_file: data_eval.jsonl
```

### Step 3: Train Model
```bash
# Verify configuration
uv run python cli.py setup strategies.yaml --strategy my_strategy --verify-only

# Run training
uv run python cli.py train --strategy my_strategy --strategy-file strategies.yaml
```

### Step 4: Evaluate Results
Training automatically shows:
- Training loss per step
- Evaluation loss every N steps
- Best checkpoint selection
- Final metrics summary

### Step 5: Deploy Model
```bash
# Convert to Ollama format
uv run python cli.py convert ./fine_tuned_models/output ./my-model --format ollama

# Test deployed model
ollama run my-model "Test prompt"
```

## Parameter Selection Guide

### Model Size vs Hardware

| Model Size | Min VRAM | Recommended | LoRA Rank | Batch Size |
|------------|----------|-------------|-----------|------------|
| 1B | 4GB | 8GB | 4-8 | 1-2 |
| 3B | 8GB | 16GB | 8-16 | 2-4 |
| 7B | 16GB | 24GB | 16-32 | 4-8 |
| 13B | 24GB | 40GB | 32-64 | 4-8 |

### Learning Rate by Dataset Size

| Dataset Size | Learning Rate | Epochs | Warmup |
|--------------|--------------|--------|--------|
| <100 | 1e-5 to 5e-5 | 5-10 | 10% |
| 100-1000 | 5e-5 to 1e-4 | 3-5 | 5% |
| 1000-10000 | 1e-4 to 2e-4 | 2-3 | 3% |
| >10000 | 2e-4 to 5e-4 | 1-2 | 1% |

### Evaluation Split Recommendations

| Use Case | Eval % | When to Use |
|----------|--------|-------------|
| Very Small Dataset (<50) | 5% | Maximum training data |
| Standard Training | 10% | Balanced approach |
| Robust Validation | 15% | Higher confidence |
| Research/Publishing | 20% | Maximum validation |

## Common Training Scenarios

### Scenario 1: Small Dataset, Prevent Overfitting
```yaml
config:
  method:
    r: 4          # Low rank
    dropout: 0.15 # High dropout
  training_args:
    num_train_epochs: 2  # Few epochs
    learning_rate: 0.00005  # Low LR
    weight_decay: 0.01  # Regularization
    eval_strategy: "steps"
    eval_steps: 20  # Frequent evaluation
```

### Scenario 2: Large GPU, Maximum Performance
```yaml
config:
  hardware:
    device: cuda
    precision:
      use_bf16: true
  method:
    r: 32  # Higher rank
  training_args:
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 1
    mixed_precision: bf16
```

### Scenario 3: Apple Silicon, Stable Training
```yaml
config:
  hardware:
    device: mps
    precision:
      use_fp16: false  # FP32 for stability
  training_args:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 2
```

## Monitoring & Debugging

### What to Watch During Training

1. **Loss Curves**
   - Training loss should decrease
   - Eval loss should follow training loss
   - Gap indicates overfitting

2. **Learning Rate**
   - Should follow schedule (warmup → decay)
   - Too high: loss spikes
   - Too low: no progress

3. **GPU Utilization**
   - Should be >80% for efficiency
   - Low utilization: increase batch size
   - OOM: decrease batch size or enable checkpointing

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Overfitting | Eval loss increases | Reduce epochs, add dropout |
| Underfitting | High train & eval loss | Increase epochs, LR, or rank |
| OOM Error | CUDA out of memory | Reduce batch size, enable checkpointing |
| Slow Training | <50% GPU usage | Increase batch size, optimize data loading |
| Unstable Loss | Spikes and jumps | Reduce learning rate, add warmup |

## Advanced Features

### Multi-GPU Training
```yaml
training_args:
  ddp_find_unused_parameters: false
  dataloader_num_workers: 4
  fp16_backend: "auto"
```

### Custom Metrics
```yaml
training_args:
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  load_best_model_at_end: true
```

### Experiment Tracking
```yaml
monitoring:
  track_training_metrics: true
  log_tensorboard: true
  save_checkpoints: true
```

## Best Practices

### ✅ DO
- Always use train/eval splits
- Start with conservative parameters
- Monitor evaluation metrics
- Save best checkpoint, not last
- Test on holdout data after training
- Document your configuration

### ❌ DON'T
- Train without evaluation
- Use very high learning rates
- Ignore overfitting signs
- Train for too many epochs on small data
- Skip data quality checks
- Forget to set random seeds

## Resources

### Documentation
- [PyTorch Training Guide](./PYTORCH_TRAINING_GUIDE.md) - Complete parameter reference
- [Strategy Examples](./STRATEGY_EXAMPLES.md) - Pre-built configurations
- [CLI Reference](./CLI.md) - Command documentation

### Demo Scripts
- `demo3_training.py` - Complete training pipeline with evaluation
- `create_data_split.py` - Data splitting utility
- `strategies.yaml` - Example strategy configurations

### External Resources
- [Hugging Face Training Docs](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PyTorch Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

*Last Updated: 2024*
*LlamaFarm Version: 2.0*