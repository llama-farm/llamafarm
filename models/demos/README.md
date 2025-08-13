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
- 📊 Cost analysis (67% savings!)

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

This directory contains **4 core demonstrations** showcasing the key capabilities of the LlamaFarm models system. Each demo uses **real API calls**, **actual model responses**, and **strategy-based configurations** with **NO SIMULATION**.

## 🚀 Demo Overview

| Demo | Use Case | Key Features | Configuration | Duration |
|------|----------|--------------|---------------|----------|
| **Demo 1** | Cloud API with Fallback | OpenAI → Ollama fallback, cost tracking | `hybrid_with_fallback` | 10-15 sec |
| **Demo 2** | Multi-Model Cloud | Task-optimized model selection | `multi_cloud` | 30-60 sec |
| **Demo 3** | Quick Training | Medical Q&A fine-tuning | `m1_fine_tuning` | 2-5 min |
| **Demo 4** | Advanced Training | Code generation fine-tuning | `code_assistant` | 5-10 min |

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

3. **For Training Demos (3 & 4)**:
   - PyTorch installed: `uv add torch transformers peft datasets`
   - 4-8GB RAM available
   - M1/M2 Mac or CUDA GPU (or CPU with patience)

### Running Individual Demos

All demos MUST be run with UV to ensure dependencies are available:

```bash
cd demos

# Demo 1: Cloud API with Local Fallback
uv run python demo1_cloud_with_fallback.py

# Demo 2: Multi-Model Cloud Strategy  
uv run python demo2_multi_model_cloud.py

# Demo 3: Quick Training Demo
uv run python demo3_quick_training.py

# Demo 4: Advanced Fine-tuning Demo
uv run python demo4_complex_training.py
```

### Running All Demos (Automated)

```bash
# Run all demos in sequence with automated responses
DEMO_MODE=automated uv run python run_all_demos.py

# Or manually with prompts
uv run python run_all_demos.py
```

## 📖 Demo Details

### Demo 1: Cloud API with Local Fallback
**File**: `demo1_cloud_with_fallback.py`  
**Strategy**: `hybrid_with_fallback`

Shows enterprise-grade reliability with automatic fallback:
- **Primary**: OpenAI GPT-4 for high-quality responses
- **Fallback**: Local Ollama models for 100% availability
- **Features**: 
  - Real-time cost tracking
  - Automatic failover handling
  - Response time monitoring
  - NO simulated responses

**Example Output**:
```
Query: What is the capital of France?
Response: The capital of France is Paris.
Source: OpenAI | Time: 1.62s | Cost: $0.0010

Query: Explain quantum computing in simple terms.
Response: Quantum computing is a type of computing that uses quantum bits...
Source: Ollama (Fallback) | Time: 5.72s | Cost: $0.0000
```

### Demo 2: Multi-Model Cloud Strategy
**File**: `demo2_multi_model_cloud.py`  
**Strategy**: Multiple models for different tasks

Demonstrates intelligent model selection by task type:
- **Simple queries**: GPT-3.5 Turbo (cost-effective)
- **Reasoning tasks**: GPT-4o Mini (balanced)
- **Creative writing**: GPT-4o (high quality)
- **Technical docs**: GPT-4 Turbo (maximum accuracy)
- **Code generation**: GPT-4o (specialized)

Shows **full responses** from each model with cost optimization metrics.

### Demo 3: Quick Training Demo
**File**: `demo3_quick_training.py`  
**Strategy**: Platform-optimized fine-tuning

Real fine-tuning workflow using strategy configurations:
- **Dataset**: Medical Q&A examples (created on-the-fly)
- **Strategies**: 
  - `m1_fine_tuning` for M1/M2 Macs
  - `cuda_fine_tuning` for NVIDIA GPUs
  - `cpu_fine_tuning` for CPU-only systems
- **Process**:
  1. Creates training dataset
  2. Selects optimal strategy for your hardware
  3. Runs actual training via CLI
  4. Shows real training progress
  5. Tests the fine-tuned model

### Demo 4: Advanced Fine-tuning Demo
**File**: `demo4_complex_training.py`  
**Strategy**: `code_assistant` with fallbacks

Advanced multi-stage training for code generation:
- **Dataset**: Code generation examples (Python functions)
- **Workflow**:
  1. Platform detection and strategy selection
  2. Multi-stage training plan
  3. Real training execution
  4. Model evaluation with actual code generation
  5. Export and deployment options

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

### Key Strategies Used:

```yaml
# For Demo 1 & 2
hybrid_with_fallback:
  cloud_api:
    type: "openai"
    default_model: "gpt-4"
  fallback_chain: "general_chain"

# For Demo 3 & 4  
m1_fine_tuning:
  fine_tuner:
    type: "pytorch"
    config:
      device: "mps"  # M1/M2 optimization
      training_args:
        per_device_train_batch_size: 1
        fp16: false  # M1 doesn't support fp16
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
- **Medical**: DeepSeek-R1-Medicalai (via Ollama)
- **Code**: CodeLlama, DeepSeek-Coder, StarCoder
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

# Step 6: Run training with demo3_training strategy
# The strategy in demos/strategies.yaml defines all parameters
uv run python cli.py train --strategy demo3_training --dataset demos/datasets/medical/medical_qa.jsonl --verbose --epochs 1 --batch-size 2

# Step 7: Convert to Ollama format (after training completes)
# Note: The convert command takes input_path output_path (positional arguments)
uv run python cli.py convert ./fine_tuned_models/medical/final_model/ ./medical-llama3.2 --format ollama --model-name medical-llama3.2

# Alternative: Convert to GGUF format with quantization
uv run python cli.py convert ./fine_tuned_models/medical ./medical-model.gguf --format gguf --quantization q4_0

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
├── demo1_cloud_with_fallback.py # Cloud + fallback demo
├── demo2_multi_model_cloud.py   # Multi-model optimization
├── demo3_quick_training.py      # Basic fine-tuning
├── demo4_complex_training.py    # Advanced training
├── run_all_demos.py            # Run all demos in sequence
├── .env.example                # Example environment variables
└── sample_datasets/            # Generated during demos
    ├── sample_medical_dataset.jsonl
    └── sample_code_dataset.jsonl
```

## ⚡ Quick Start

1. **Fastest Demo** (10 seconds):
```bash
uv run python demo1_cloud_with_fallback.py
```

2. **Most Visual** (shows full responses):
```bash
uv run python demo2_multi_model_cloud.py
```

3. **Most Educational** (explains training):
```bash
uv run python demo3_quick_training.py
```

4. **Most Comprehensive** (all features):
```bash
DEMO_MODE=automated uv run python run_all_demos.py
```

## 🐛 Troubleshooting

### "Unknown cloud_apis type: openai"
Run with UV: `uv run python demo1_cloud_with_fallback.py`

### "Strategy not found: m1_fine_tuning"
The strategy exists but wasn't loaded. This is fixed in the latest version.

### "No module named 'tiktoken'"
Already added to dependencies. Run `uv sync` to install.

### Training demos timeout
Training can take time. The demos show the process but may not complete full training in the timeout period. This is normal.

## 💡 Next Steps

1. **Customize Strategies**: Modify `default_strategies.yaml` for your needs
2. **Create New Datasets**: Add your domain-specific training data
3. **Try Different Models**: Explore the 40+ models in the catalog
4. **Production Deployment**: Use the trained models in your applications
5. **Contribute**: Share your strategies and improvements

---

**🎯 Ready to see LlamaFarm in action? Start with Demo 1 for instant results!**