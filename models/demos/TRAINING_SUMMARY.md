# Training and Fine-tuning in LlamaFarm Models

## Overview

All training and fine-tuning in LlamaFarm is accomplished through:
1. **Strategy configurations** in `default_strategies.yaml`
2. **CLI commands** that use these strategies
3. **No hardcoded values** in demo scripts

## How Training Works

### 1. Strategy-Based Configuration

All training settings are defined in strategies:

```yaml
# From default_strategies.yaml
m1_fine_tuning:
  description: "M1/M2 Mac optimized fine-tuning with MPS backend"
  fine_tuner:
    type: "pytorch"
    config:
      base_model:
        name: "llama3.2-3b"
        huggingface_id: "meta-llama/Llama-3.2-3B"
      method:
        type: "lora"
        r: 8  # Smaller for M1
        alpha: 16
        dropout: 0.1
        target_modules: ["q_proj", "v_proj"]
      training_args:
        num_train_epochs: 2
        per_device_train_batch_size: 1
        learning_rate: 3e-4
        device: "mps"  # M1 Metal Performance Shaders
```

### 2. CLI Commands for Training

Training is initiated through CLI commands, not direct code:

```bash
# List available strategies
python cli.py finetune strategies list

# Estimate resource requirements
python cli.py finetune estimate --strategy m1_fine_tuning

# Start training with a strategy
python cli.py finetune start --strategy m1_fine_tuning --dataset data.jsonl

# Monitor training progress
python cli.py finetune monitor --job-id <job-id>

# Export trained model
python cli.py finetune export --model-path ./fine_tuned_models/model/ --format ollama
```

### 3. Demo Scripts Use CLI Commands

Demo scripts demonstrate training by calling CLI commands:

```python
# From demo3_quick_training.py
def run_cli_command(command: str, show_output: bool = True):
    """Run a CLI command and capture output."""
    cmd_parts = command.split()
    result = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result.stdout, result.stderr

# Start training with strategy
output, error = run_cli_command(
    f"python cli.py finetune start --strategy {strategy} --dataset {dataset_path}"
)
```

## Available Training Strategies

### 1. Platform-Specific Strategies

- **m1_fine_tuning**: Optimized for M1/M2 Macs with MPS backend
- **cuda_fine_tuning**: For NVIDIA GPUs with CUDA
- **cpu_fine_tuning**: For systems without GPU
- **llamafactory_advanced**: Cross-platform with LlamaFactory

### 2. Model-Specific Training

Each strategy defines:
- Base model and HuggingFace ID
- Fine-tuning method (LoRA, QLoRA, full)
- Training hyperparameters
- Device configuration
- Memory optimizations

## Demo Scripts

### Demo 3: Quick Training
- Shows basic fine-tuning workflow
- Uses platform-appropriate strategy
- Creates sample medical Q&A dataset
- Demonstrates training progress monitoring

### Demo 4: Advanced Fine-tuning
- Multi-stage training process
- Code generation dataset
- Model evaluation and testing
- Export and deployment options

## Key Benefits

1. **No Hardcoded Values**: All configuration in strategy files
2. **Platform Optimization**: Automatic selection based on hardware
3. **Reproducibility**: Same strategy = same results
4. **Easy Switching**: Change strategies without code changes
5. **Version Control**: Strategy files can be tracked in git

## Training Process Flow

```
1. Select Strategy → 2. Prepare Dataset → 3. Validate Config
        ↓                                           ↓
4. Start Training → 5. Monitor Progress → 6. Evaluate Model
        ↓                                           ↓
7. Export Model → 8. Deploy (Ollama/API) → 9. Use Fine-tuned Model
```

## Example Training Session

```bash
# 1. Check available strategies
$ python cli.py finetune strategies list

# 2. Create/prepare your dataset
$ echo '{"instruction": "...", "output": "..."}' > data.jsonl

# 3. Start training with M1 strategy
$ python cli.py finetune start --strategy m1_fine_tuning --dataset data.jsonl

# 4. Monitor progress
$ python cli.py finetune monitor --job-id abc123

# 5. Export to Ollama
$ python cli.py finetune export --model-path ./fine_tuned_models/model/ --format ollama

# 6. Use the model
$ python cli.py generate --model ./fine_tuned_models/model/ --prompt "Test prompt"
```

## Configuration Through Strategies Only

All training parameters come from strategy files:
- Model selection
- Training hyperparameters
- Device configuration
- Optimization settings
- Memory management
- Fallback options

No training parameters should be hardcoded in application code!