# LlamaFarm Models - Demo Scripts

## Overview

These are **standalone Python scripts** that demonstrate LlamaFarm Models capabilities using real CLI commands. No hardcoding - everything is transparent and educational!

## 🎯 Available Demos

### Demo 1: Cloud Model Fallback
**File:** `demo1_cloud_fallback.py`

Shows automatic failover from cloud APIs to local models when services are unavailable.

```bash
python demos/demo1_cloud_fallback.py
```

**What you'll see:**
- ✅ Successful cloud API calls
- ❌ Simulated API failure  
- 🔄 Explanation of fallback chains
- 🦙 Local model alternatives

### Demo 2: Multi-Model Optimization
**File:** `demo2_multi_model.py`

Demonstrates intelligent task routing for cost and performance optimization.

```bash
python demos/demo2_multi_model.py
```

**What you'll see:**
- 💨 Simple queries → Fast, cheap models
- 🧠 Complex tasks → Advanced models
- 🎨 Creative tasks → Specialized models
- 💻 Code tasks → Local models (free!)
- 📊 Cost analysis (60-80% savings!)

### Demo 3: Training Pipeline
**File:** `demo3_training.py`

Complete fine-tuning pipeline with evaluation and before/after comparison.

```bash
python demos/demo3_training.py
```

**What you'll see:**
- 📝 Base model performance
- 📊 90/10 train/eval data split
- 🏋️ Training progress with evaluation metrics
- ✨ Fine-tuned improvements
- 🦙 Ollama conversion
- 📈 Best model selection based on eval loss

### Run All Demos
**File:** `run_all_demos.py`

Runs all three demos in sequence.

```bash
# Interactive mode
python demos/run_all_demos.py

# Automated mode (for CI/CD)
DEMO_MODE=automated python demos/run_all_demos.py --auto --quick
```

## 🚀 Key Features

### Real CLI Commands
Every demo runs actual CLI commands that you can see:
```bash
$ uv run python cli.py complete "What is 2+2?" --strategy demo2_multi_model --strategy-file demos/strategies.yaml
```

### Educational Flow
- Step-by-step explanations
- "Press Enter to continue" prompts
- Cost analysis and comparisons
- Real performance metrics
- **NEW**: Evaluation metrics during training

### Transparent Implementation
- No hidden abstractions
- All commands visible
- Copy/paste to try yourself
- Strategy-driven configuration
- **NEW**: Train/eval data splits for robust training

This directory contains **3 core demonstrations** showcasing the key capabilities of the LlamaFarm models system. Each demo uses **real API calls**, **actual model responses**, and **strategy-based configurations** with **NO SIMULATION**.

## 🎭 Important: NO SIMULATION

All demos make **real API calls** and show **actual responses**. There is **zero simulation code** in these demos. What you see is what the models actually generate.

## 📋 Running the Demos

### Prerequisites

1. **Environment Setup**:
```bash
# Install dependencies with UV
uv sync

# Copy and configure environment variables
cp demos/.env.example demos/.env
# Edit .env to add your API keys
```

2. **For Cloud Demos (1 & 2)**:
   - OpenAI API key in `.env` file
   - Ollama running locally: `ollama serve`

3. **For Training Demos (3)**:
   - PyTorch installed: `uv add torch transformers peft datasets`
   - 4-8GB RAM available
   - M1/M2 Mac or CUDA GPU (or CPU with patience)

### Running Individual Demos

All demos MUST be run with UV to ensure dependencies are available:

```bash
cd models

# Demo 1: Cloud API with Local Fallback
uv run python demos/demo1_cloud_fallback.py

# Demo 2: Multi-Model Cloud Strategy  
uv run python demos/demo2_multi_model.py

# Demo 3: Training Demo
uv run python demos/demo3_training.py
```

### Running All Demos (Automated)

```bash
# Run all demos in sequence with automated responses
DEMO_MODE=automated uv run python demos/run_all_demos.py

# Or manually with prompts
uv run python demos/run_all_demos.py
```

## 📊 Training Features (NEW)

### Evaluation During Training
- **Automatic Data Splitting**: 90/10 train/eval split by default
- **Custom Split Ratios**: 5%, 10%, 15%, or 20% for evaluation
- **Real-time Metrics**: Track train and eval loss during training
- **Best Model Selection**: Automatically saves best checkpoint
- **Overfitting Detection**: Monitor eval vs train loss gap

### Data Split Utility
```bash
# Create custom train/eval splits
python demos/create_data_split.py --input data.jsonl --eval-percent 10  # Standard
python demos/create_data_split.py --input data.jsonl --eval-percent 15  # Robust
python demos/create_data_split.py --input data.jsonl --eval-percent 5   # Maximum training
```

### Evaluation Benefits
| Feature | Benefit |
|---------|---------|
| Holdout Validation | Unbiased performance metrics |
| Early Stopping | Prevent overfitting |
| Best Model Selection | Use optimal checkpoint |
| Generalization Metrics | Confidence in deployment |

## 🔧 Configuration Through Strategies

All demos use **strategy-based configuration** from `strategies.yaml`. No hardcoded values!

### Key Strategies Used (Array Format):

```yaml
strategies:
  - name: demo1_cloud_fallback
    description: Cloud API with automatic fallback to local models
    components:
      cloud_api:
        type: openai_compatible
        config:
          model: gpt-4-turbo-preview
      ollama:
        type: ollama
        config:
          model: llama3.2:3b
    fallback_chain:
      - provider: cloud_api
        model: gpt-4-turbo-preview
      - provider: ollama
        model: llama3.2:3b

  - name: demo3_training
    description: Fine-tuning configuration  
    components:
      fine_tuner:
        type: pytorch
        config:
          base_model: meta-llama/Llama-3.2-3B-Instruct
          lora_r: 4
          lora_alpha: 16
          lora_dropout: 0.1
          learning_rate: 5e-5
          num_epochs: 2
          per_device_train_batch_size: 1
          gradient_accumulation_steps: 4
```

## 🎯 What You'll See

### Real API Responses
- Actual text generated by OpenAI models
- Real responses from local Ollama models
- Genuine fallback behavior when APIs fail
- True cost calculations based on token usage

### Real Training Progress
- Actual dataset creation
- Real strategy selection based on your hardware
- Genuine training commands executed
- Real model evaluation (when training completes)

### NO Simulation
- No fake responses
- No simulated training progress
- No dummy data
- Everything is real!

## 📊 Model Catalog Integration

The demos use models from the comprehensive model catalog:

```bash
# View available models
uv run python cli.py catalog list

# See model details
uv run python cli.py catalog info "llama3.2:3b"

# View fallback chains
uv run python cli.py catalog fallbacks --chain medical_chain
```

### Featured Models:
- **Medical**: Medical-optimized Llama models (via fine-tuning)
- **Code**: CodeLlama, Mistral (via Ollama)
- **General**: Llama 3.2, Mistral, Phi-3.5
- **Cloud**: GPT-4, GPT-3.5, Claude (when available)

## 📚 Available Strategies

The demos use various strategies defined in `demos/strategies.yaml`:

### Demo Strategies
- **`demo1_cloud_fallback`** - Cloud API with automatic fallback to local models
- **`demo2_multi_model`** - Intelligent routing to different models based on task
- **`demo3_base_model`** - Base Llama 3.2:3b model (before training)
- **`demo3_finetuned_model`** - Fine-tuned medical model (after training)
- **`demo3_training`** - Training configuration for fine-tuning
- **`demo3_training_optimized`** - Optimized training with smaller batch sizes

### Testing Strategies  
- **`test_tinyllama`** - TinyLlama 1.1B for quick tests
- **`test_mistral`** - Mistral 7B for general purpose
- **`test_codellama`** - Code Llama for programming tasks
- **`test_phi3`** - Microsoft Phi-3 Mini (3.8B)

### Environment Strategies
- **`local_development`** - Local-only models for development
- **`production_hybrid`** - Production with cloud + local fallback
- **`mock_development`** - Mock responses for testing

### Hardware Training Strategies
- **`training_mps_apple`** - Optimized for Apple Silicon
- **`training_cuda_consumer`** - For NVIDIA consumer GPUs
- **`training_cuda_datacenter`** - For datacenter GPUs
- **`training_cpu_only`** - CPU-only training

## 🔐 Provider-Agnostic Completions (NEW)

The CLI now supports provider-agnostic completions using the `complete` command. This abstracts away whether you're using Ollama, OpenAI, or any other provider - the strategy determines where the request is routed.

### Why Provider-Agnostic?
- **Flexibility**: Switch providers without changing code
- **Abstraction**: Users don't need to know if it's local or cloud
- **Strategy-driven**: Let the strategy decide the best provider
- **Unified interface**: Same command works for all providers

### Usage Examples

```bash
# Instead of provider-specific commands like:
# OLD: uv run python cli.py ollama run llama3.2:3b "prompt"
# OLD: uv run python cli.py query "prompt" --provider openai 

# Test different models using strategies:
uv run python cli.py complete "Medical question" --strategy demo3_base_model --strategy-file demos/strategies.yaml  # Base Llama 3.2
uv run python cli.py complete "Medical question" --strategy demo3_finetuned_model --strategy-file demos/strategies.yaml  # After fine-tuning
uv run python cli.py complete "Quick test" --strategy test_tinyllama --strategy-file demos/strategies.yaml  # Fast 1.1B model
uv run python cli.py complete "Write code" --strategy test_codellama --strategy-file demos/strategies.yaml  # Code-specific
uv run python cli.py complete "General query" --strategy test_mistral --strategy-file demos/strategies.yaml  # Mistral 7B

# With custom strategy file:
uv run python cli.py complete "Your prompt here" \
  --strategy production_hybrid \
  --strategy-file configs/production.yaml

# With options:
uv run python cli.py complete "Explain quantum computing" \
  --strategy demo1_cloud_fallback \
  --strategy-file demos/strategies.yaml \
  --temperature 0.7 \
  --max-tokens 500 \
  --verbose  # Shows which provider is actually used

# With system prompt:
uv run python cli.py complete "What are the symptoms?" \
  --strategy demo3_training \
  --strategy-file demos/strategies.yaml \
  --system "You are a medical assistant. Be concise."

# Using a completely different strategy file:
uv run python cli.py complete "Write code to parse JSON" \
  --strategy-file my_strategies.yaml \
  --strategy code_generation
```

The strategy determines:
- Which provider to use (Ollama, OpenAI, etc.)
- Which model to use
- Fallback chains if primary fails
- All configuration details

## 🚀 CLI Commands Used in Demos

### Demo 1: Cloud Fallback - Command by Command

```bash
# Step 1: Setup requirements
uv run python cli.py setup demos/strategies.yaml --auto

# Step 2: Test cloud provider (OpenAI)
uv run python cli.py test --strategy demo1_cloud_fallback --provider cloud_api

# Step 3: Query with cloud provider (provider-agnostic)
uv run python cli.py complete "What is the capital of France?" --strategy demo1_cloud_fallback --strategy-file demos/strategies.yaml

# Step 4: Simulate fallback to Ollama
# First ensure Ollama is running
uv run python cli.py ollama status

# List local models
uv run python cli.py ollama list

# Test local model (provider-agnostic)
uv run python cli.py complete "Test query" --strategy demo1_cloud_fallback --strategy-file demos/strategies.yaml --verbose

# Step 5: Test fallback chain
uv run python cli.py test --strategy demo1_cloud_fallback --test-fallback
```

### Demo 2: Multi-Model - Command by Command

```bash
# Step 1: Setup
uv run python cli.py setup demos/strategies.yaml --strategy demo2_multi_model --auto

# Step 2: Test different task types

# Simple query (provider-agnostic)
uv run python cli.py complete "What is 2+2?" --strategy demo2_multi_model --strategy-file demos/strategies.yaml

# Complex reasoning
uv run python cli.py complete "Explain quantum computing" --strategy demo2_multi_model --strategy-file demos/strategies.yaml

# Creative writing
uv run python cli.py complete "Write a haiku about programming" --strategy demo2_multi_model --strategy-file demos/strategies.yaml

# Code generation
uv run python cli.py complete "Write a Python fibonacci function" --strategy demo2_multi_model --strategy-file demos/strategies.yaml

# Step 3: Compare costs
uv run python cli.py analyze costs --strategy demo2_multi_model
```

### Demo 3: Training Pipeline - Command by Command

These are the EXACT commands used in demo3_training.py:

```bash
# Step 1: Setup training requirements (includes converters)
uv run python cli.py setup demos/strategies.yaml --auto --verbose

# Step 2: Check Ollama status
uv run python cli.py ollama status

# Step 3: List installed Ollama models
uv run python cli.py ollama list

# Step 4: Pull Llama 3.2:3b if not present
uv run python cli.py ollama pull llama3.2:3b

# Step 5: Test base model before training (using base model strategy)
uv run python cli.py complete "What are the symptoms of diabetes?" --strategy demo3_base_model --strategy-file demos/strategies.yaml
uv run python cli.py complete "How do you treat hypertension?" --strategy demo3_base_model --strategy-file demos/strategies.yaml
uv run python cli.py complete "What are the side effects of statins?" --strategy demo3_base_model --strategy-file demos/strategies.yaml

# Step 6: Run training with demo3_training_optimized strategy (smaller batch size)
# The strategy saves to: ./fine_tuned_models/medical_optimized/
uv run python cli.py train --strategy demo3_training_optimized --dataset demos/datasets/medical/medical_qa_cleaned.jsonl --verbose --epochs 2 --batch-size 1

# Step 7: Convert to Ollama format (after training completes)
# The training creates a LoRA adapter (small, ~50MB) in ./fine_tuned_models/medical_optimized/final_model/
# LoRA needs the base model for conversion

# First, ensure base model is cached locally (downloads ~7GB if not cached)
uv run python cli.py download meta-llama/Llama-3.2-3B-Instruct

# Then convert - the converter will automatically merge LoRA with base model
uv run python cli.py convert ./fine_tuned_models/medical_optimized/final_model/ ./medical-llama3.2-optimized --format ollama --model-name medical-llama3.2-optimized

# Note: If using demo3_training (not optimized) strategy, paths would be:
#   ./fine_tuned_models/medical/final_model/ (LoRA adapter)
#   ./fine_tuned_models/medical/checkpoints/ (training checkpoints)

# Step 8: Test fine-tuned model (using finetuned model strategy)
uv run python cli.py complete "What are the symptoms of diabetes?" --strategy demo3_finetuned_model --strategy-file demos/strategies.yaml
```

### Hardware-Specific Training Commands

```bash
# For Apple Silicon (M1/M2/M3)
uv run python cli.py train \
  --strategy training_mps_apple \
  --dataset demos/datasets/medical/medical_qa.jsonl \
  --device mps \
  --batch-size 2

# For NVIDIA GPU
uv run python cli.py train \
  --strategy training_cuda_consumer \
  --dataset demos/datasets/medical/medical_qa.jsonl \
  --device cuda \
  --fp16 \
  --batch-size 4

# For CPU only
uv run python cli.py train \
  --strategy training_cpu_only \
  --dataset demos/datasets/medical/medical_qa.jsonl \
  --device cpu \
  --batch-size 1
```

### Mock Development Commands (for testing without resources)

```bash
# Setup mock strategy
uv run python cli.py setup demos/strategies.yaml --strategy mock_development --auto

# Test with mock model
uv run python cli.py complete "What is machine learning?" --strategy mock_development --strategy-file demos/strategies.yaml

# Run all test queries
uv run python cli.py test --strategy mock_development --all
```

### Utility Commands

```bash
# List all available strategies
uv run python cli.py strategies list

# Show strategy details
uv run python cli.py strategies show demo3_training

# Validate strategy configuration
uv run python cli.py strategies validate demos/strategies.yaml

# Check system requirements
uv run python cli.py check-requirements --strategy demo3_training

# Monitor resource usage during training
uv run python cli.py monitor --pid <training-process-id>

# Clean up temporary files
uv run python cli.py cleanup --temp-files --cache
```

### Running Complete Demos

```bash
# Run demo 1 (Cloud Fallback)
uv run python demos/demo1_cloud_fallback.py

# Run demo 2 (Multi-Model) 
uv run python demos/demo2_multi_model.py

# Run demo 3 (Training)
uv run python demos/demo3_training.py

# Run all demos automated
DEMO_MODE=automated uv run python demos/run_all_demos.py --auto

# Run specific demo automated
DEMO_MODE=automated timeout 60 uv run python demos/demo1_cloud_fallback.py
```

## 🎓 Learning Objectives

After running the demos, you'll understand:

1. **Strategy-Based Configuration**: How strategies simplify complex setups
2. **Fallback Chains**: Building reliable AI systems with multiple fallbacks
3. **Cost Optimization**: Selecting appropriate models for different tasks
4. **Real Training**: How fine-tuning actually works with modern tools
5. **Platform Optimization**: Leveraging M1/MPS, CUDA, or CPU effectively
6. **Production Patterns**: Building reliable, cost-effective AI applications

## 📁 Directory Structure

```
demos/
├── README.md                    # This file
├── demo1_cloud_fallback.py     # Cloud + fallback demo
├── demo2_multi_model.py         # Multi-model optimization
├── demo3_training.py            # Medical fine-tuning
├── demo_mock_model.py           # Mock model testing
├── run_all_demos.py            # Run all demos in sequence
├── create_data_split.py        # Dataset splitting utility
├── strategies.yaml             # All strategy configurations
├── TRAINING_SUMMARY.md         # Training implementation details
├── .env.example                # Example environment variables
└── datasets/                   # Training datasets
    ├── medical/                # Medical Q&A datasets
    │   ├── medical_qa.jsonl
    │   ├── medical_qa_train.jsonl
    │   ├── medical_qa_eval.jsonl
    │   └── medical_qa_cleaned.jsonl
    ├── code_helper/            # Programming datasets
    ├── creative_writing/       # Creative content
    ├── customer_support/       # Support examples
    └── technical_qa/           # Engineering Q&A
```

## ⚡ Quick Start

1. **Fastest Demo** (10 seconds):
```bash
uv run python demos/demo1_cloud_fallback.py
```

2. **Most Visual** (shows full responses):
```bash
uv run python demos/demo2_multi_model.py
```

3. **Most Educational** (explains training):
```bash
uv run python demos/demo3_training.py
```

4. **Most Comprehensive** (all features):
```bash
DEMO_MODE=automated uv run python demos/run_all_demos.py
```

## 🐛 Troubleshooting

### "Strategy not found"
The strategies now use array format. Ensure your `strategies.yaml` has:
```yaml
strategies:
  - name: strategy_name
    # ... configuration
```

### "No module named 'tiktoken'"
Run `uv sync` to install all dependencies.

### Training demos timeout
Training can take time. The demos show the process but may not complete full training in the timeout period. This is normal.

### Ollama not responding
```bash
# Start Ollama
ollama serve

# Check status
uv run python cli.py ollama status

# Pull model
ollama pull llama3.2:3b
```

### "Conversion failed: config.json not found"
This happens when trying to convert a LoRA adapter. The training produces a LoRA adapter (50MB) not a full model (7GB).

**Solution:**
```bash
# 1. Download the base model first (required for LoRA conversion)
uv run python cli.py download meta-llama/Llama-3.2-3B-Instruct

# 2. Use the correct path with /final_model/ at the end
# For demo3_training_optimized:
uv run python cli.py convert ./fine_tuned_models/medical_optimized/final_model/ ./medical-llama3.2 --format ollama --model-name medical-llama3.2

# For demo3_training:
uv run python cli.py convert ./fine_tuned_models/medical/final_model/ ./medical-llama3.2 --format ollama --model-name medical-llama3.2
```

The converter will automatically merge the LoRA adapter with the base model during conversion.

## 💡 Next Steps

1. **Customize Strategies**: Modify `demos/strategies.yaml` for your needs
2. **Create New Datasets**: Add your domain-specific training data
3. **Try Different Models**: Explore the 40+ models in the catalog
4. **Production Deployment**: Use the trained models in your applications
5. **Contribute**: Share your strategies and improvements

---

**🎯 Ready to see LlamaFarm in action? Start with Demo 1 for instant results!**