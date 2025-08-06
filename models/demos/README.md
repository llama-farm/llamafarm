# 🎯 LlamaFarm Models Demo Showcase

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

## 🔧 Configuration Through Strategies

All demos use **strategy-based configuration** from `default_strategies.yaml`. No hardcoded values!

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

## 🚀 CLI Commands Used in Demos

The demos execute real CLI commands:

```bash
# List available fine-tuning strategies
python cli.py finetune strategies list

# Estimate resource requirements
python cli.py finetune estimate --strategy m1_fine_tuning

# Start fine-tuning
python cli.py finetune start --strategy m1_fine_tuning --dataset dataset.jsonl

# Monitor training
python cli.py finetune monitor --job-id training-job-123

# Generate with fine-tuned model
python cli.py generate --model ./fine_tuned_models/model --prompt "Query"
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