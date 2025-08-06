# 🦙 LlamaFarm Models System

> **Unified Model Management for Cloud & Local LLMs** - Complete setup guide, usage examples, and training workflows

A comprehensive model management system providing unified access to **25+ cloud and local LLMs** with real API integration, fallback chains, and production-ready features.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for cloud providers (optional)
- [Ollama](https://ollama.ai) for local models (optional)

### 1. Installation
```bash
# Clone and navigate to models directory
cd llamafarm-1/models

# Install dependencies
uv sync

# Copy environment template
cp ../.env.example ../.env
# Edit ../.env with your API keys
```

### 2. Basic Usage
```bash
# List available models
uv run python cli.py list

# Send your first query
uv run python cli.py query "What is machine learning?"

# Start interactive chat
uv run python cli.py chat

# Use a specific model
uv run python cli.py query "Explain quantum computing" --provider openai_gpt4_turbo

# Use different configuration files (YAML or JSON supported)
uv run python cli.py --config config/development.yaml list
uv run python cli.py --config config/production.yaml query "Production query"
uv run python cli.py --config config/ollama_local.yaml chat
```

### 3. Run Complete Demo
```bash
# Automated setup and comprehensive demo
./setup_and_demo.sh
```

## 🤖 Supported Providers & Models

### ☁️ **Cloud Providers**
| Provider | Models | Key Features |
|----------|---------|--------------|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo | Industry-leading performance, fast responses |
| **Anthropic** | Claude 3 Opus, Sonnet, Haiku | Excellent reasoning, large context windows |
| **Together AI** | Llama 3.1 70B, Mixtral 8x7B, Code Llama | Open-source models, competitive pricing |
| **Groq** | Llama 3 70B, Mixtral 8x7B | Ultra-fast inference (500+ tokens/sec) |
| **Cohere** | Command R+, Command R | Enterprise-focused, RAG optimization |

### 🏠 **Local Providers**
| Provider | Models | Key Features |
|----------|---------|--------------|  
| **Ollama** | Llama 3.1/3.2, Mistral, Phi-3, CodeLlama | Easy setup, no API costs, privacy |
| **Hugging Face** | GPT-2, DistilGPT-2, custom models | Open ecosystem, custom fine-tuning |
| **vLLM** | Llama 2/3, Mistral, CodeLlama | High-throughput local inference |
| **TGI** | Any HF model | Production deployment, batching |

## 📋 Complete Feature Guide

### 🎯 **Core Commands**

#### **Query - Send Single Requests**
```bash
# Basic query
uv run python cli.py query "Explain quantum computing"

# Use specific provider
uv run python cli.py query "Write Python code" --provider openai_gpt4o_mini

# Control generation parameters
uv run python cli.py query "Tell a creative story" --temperature 0.9 --max-tokens 500

# Add system prompt
uv run python cli.py query "Analyze this data" --system "You are a data scientist"

# Stream response in real-time
uv run python cli.py query "Tell a long story" --stream

# Save response to file
uv run python cli.py query "Write a README" --save output.md

# Output as JSON
uv run python cli.py query "List AI facts" --json
```

#### **Chat - Interactive Sessions**
```bash
# Start basic chat
uv run python cli.py chat

# Chat with specific model
uv run python cli.py chat --provider anthropic_claude_3_haiku

# Set system prompt for session
uv run python cli.py chat --system "You are a coding assistant"

# Save chat history
uv run python cli.py chat --save-history my_session.json

# Load previous chat
uv run python cli.py chat --history previous_session.json
```

#### **Send - File Analysis**
```bash
# Send code for review
uv run python cli.py send code.py --prompt "Review this code"

# Analyze documents
uv run python cli.py send document.txt --prompt "Summarize key points"

# Process data files
uv run python cli.py send data.csv --prompt "Analyze trends"

# Save analysis to file
uv run python cli.py send script.js --output analysis.md
```

#### **Batch - Multiple Queries**
```bash
# Process queries from file
echo -e "What is AI?\nExplain ML\nDefine NLP" > queries.txt
uv run python cli.py batch queries.txt

# Use specific provider
uv run python cli.py batch queries.txt --provider openai_gpt4o_mini

# Parallel processing
uv run python cli.py batch large_file.txt --parallel 5

# Save all responses
uv run python cli.py batch questions.txt --output responses.json
```

### 🔧 **Management Commands**

#### **Provider Management**
```bash
# List configured providers
uv run python cli.py list

# Detailed view with costs
uv run python cli.py list --detailed

# Test specific provider
uv run python cli.py test openai_gpt4o_mini

# Health check all providers
uv run python cli.py health-check

# Compare responses
uv run python cli.py compare --providers openai_gpt4o_mini,anthropic_claude_3_haiku --query "Explain recursion"
```

#### **Configuration Management** 
```bash
# Validate current config
uv run python cli.py validate-config

# Generate basic config
uv run python cli.py generate-config --type basic

# Generate production config
uv run python cli.py generate-config --type production --output prod.json

# Use custom config
uv run python cli.py --config custom.json list
```

### 🏠 **Local Model Management**

#### **Ollama Integration**
```bash
# List local Ollama models
uv run python cli.py list-local

# Pull new models
uv run python cli.py pull llama3.2:3b
uv run python cli.py pull mistral:latest

# Test local models
uv run python cli.py test-local llama3.1:8b

# Generate Ollama config
uv run python cli.py generate-ollama-config --output ollama.json
```

#### **Hugging Face Integration**
```bash
# Login to HF Hub
uv run python cli.py hf-login

# Search models
uv run python cli.py list-hf --search "gpt2" --limit 10

# Download models
uv run python cli.py download-hf gpt2
uv run python cli.py download-hf distilbert-base-uncased --cache-dir ./models

# Test HF models
uv run python cli.py test-hf gpt2 --query "Once upon a time"

# Generate HF config
uv run python cli.py generate-hf-config --models "gpt2,distilgpt2"
```

#### **High-Performance Inference**
```bash
# List vLLM compatible models
uv run python cli.py list-vllm

# Test vLLM inference
uv run python cli.py test-vllm meta-llama/Llama-2-7b-chat-hf --query "Hello"

# Test TGI endpoints
uv run python cli.py test-tgi --endpoint http://localhost:8080 --query "Test"

# Generate engines config
uv run python cli.py generate-engines-config --output engines.json
```

## ⚙️ Configuration Guide

### 🎛️ **Configuration Format**
The models system supports both **YAML** (recommended) and **JSON** configuration files. YAML provides better readability and maintainability.

### **Basic Configuration**
```yaml
name: "My AI Models"
version: "1.0.0"
default_provider: "openai_gpt4o_mini"

providers:
  openai_gpt4o_mini:
    type: "cloud"
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
    max_tokens: 2048
```

### **Available Configuration Files**
- **`config/default.yaml`** - Comprehensive template with all options
- **`config/development.yaml`** - Optimized for local development
- **`config/production.yaml`** - Production-ready with robust fallbacks
- **`config/ollama_local.yaml`** - Complete local models configuration via Ollama
- **`config/use_case_examples.yaml`** - 8 specialized use case configurations (customer support, code generation, content creation, data analysis, real-time chat, privacy-first, multilingual, cost-optimized)

### 🔄 **Production Configuration with Fallbacks**
```yaml
name: "Production Setup"
default_provider: "primary"
fallback_chain: 
  - "primary"
  - "secondary" 
  - "local_backup"

providers:
  primary:
    type: "cloud"
    provider: "openai"
    model: "gpt-4o-mini"
    api_key: "${OPENAI_API_KEY}"
    timeout: 30

  secondary:
    type: "cloud"
    provider: "anthropic"
    model: "claude-3-haiku-20240307"
    api_key: "${ANTHROPIC_API_KEY}"
    timeout: 45

  local_backup:
    type: "local"
    provider: "ollama"
    model: "llama3.1:8b"
    host: "localhost"
```

### 🎯 **Use Case Specific Configurations**

See `config/use_case_examples.yaml` for optimized setups:
- **Customer Support**: Fast, helpful responses with cost control
- **Code Generation**: Optimized for programming with specialized models
- **Content Creation**: High creativity settings for marketing and writing
- **Data Analysis**: Analytical accuracy with advanced reasoning models
- **Real-time Chat**: Ultra-fast inference for interactive applications
- **Privacy-First**: Local-only models for sensitive data
- **Multilingual**: Optimized for international and cross-language use
- **Cost-Optimized**: Minimize expenses while maintaining quality

### 🌐 **Environment Variables**
```bash
# Required for cloud providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...

# Optional for Hugging Face
HF_TOKEN=hf_...

# Optional for local models
OLLAMA_HOST=localhost
```

## 🎓 Fine-Tuning System

> **Production Ready**: Comprehensive fine-tuning with strategy-based configuration and hardware optimization

The LlamaFarm Models system now includes a complete fine-tuning implementation with support for LoRA, QLoRA, and full fine-tuning using PyTorch and LlamaFactory.

### 🚀 **Quick Start**

#### **Strategy-Based Fine-Tuning (Recommended)**
```bash
# List available strategies
uv run python cli.py finetune strategies list

# Use a pre-configured strategy
uv run python cli.py finetune start --strategy mac_m1_lora --dataset my_data.jsonl

# Get strategy recommendations
uv run python cli.py finetune strategies recommend --hardware mac --model-size 8b

# Estimate resource requirements
uv run python cli.py finetune estimate --strategy gpu_full_finetune
```

#### **Custom Configuration**
```bash
# Start training with custom settings
uv run python cli.py finetune start \
  --dataset my_data.jsonl \
  --base-model llama3.1-8b \
  --method lora \
  --output-dir ./my_fine_tuned_model

# Dry run to validate configuration
uv run python cli.py finetune start --dataset my_data.jsonl --strategy mac_m1_lora --dry-run
```

### 📋 **Available Strategies**

| Strategy | Hardware | Model Size | Method | Use Cases |
|----------|----------|------------|--------|-----------|
| **mac_m1_lora** | Mac M1/M2 | 3B-8B | LoRA | Personal projects, coding helpers |
| **gpu_full_finetune** | High-end GPU | 8B-70B | Full/LoRA | Production models, research |
| **cpu_small_model** | CPU-only | 3B | LoRA | Testing, experimentation |
| **cloud_scalable** | Multi-GPU | 8B-70B+ | QLoRA | Large-scale training |

#### **Strategy Details**
```bash
# View detailed strategy information
uv run python cli.py finetune strategies show mac_m1_lora

# Get hardware-specific recommendations
uv run python cli.py finetune strategies recommend --hardware gpu --use-case production
```

### 🛠️ **Supported Methods**

#### **LoRA (Low-Rank Adaptation)**
- **Memory Efficient**: ~4x less memory than full fine-tuning
- **Fast Training**: Significantly faster training times
- **Good Quality**: Maintains most of the original model quality
- **Recommended for**: Personal projects, experimentation, resource-constrained environments

#### **QLoRA (Quantized LoRA)**
- **Ultra Memory Efficient**: ~16x less memory than full fine-tuning
- **4-bit Quantization**: Uses NF4 quantization for maximum efficiency
- **Large Model Support**: Fine-tune 70B+ models on consumer hardware
- **Recommended for**: Large models, limited VRAM, cloud cost optimization

#### **Full Fine-Tuning**
- **Maximum Quality**: Best possible fine-tuning results
- **High Resource Requirements**: Requires significant GPU memory
- **Slower Training**: Takes longer but provides complete model adaptation
- **Recommended for**: Production models, research, domain-specific applications

### 💻 **Hardware Compatibility**

#### **Mac Apple Silicon (M1/M2/M3)**
```bash
# Optimized for Mac with Metal Performance Shaders
uv run python cli.py finetune start --strategy mac_m1_lora --dataset data.jsonl

# Recommended settings:
# - Models: Up to 8B parameters (Llama 3.1 8B, Mistral 7B)
# - Memory: 16GB+ unified memory recommended
# - Method: LoRA with small batch sizes
# - Expected time: 2-4 hours for 1000 samples
```

#### **High-End GPUs (RTX 4090, A100, H100)**
```bash
# Full fine-tuning on powerful GPUs
uv run python cli.py finetune start --strategy gpu_full_finetune --dataset data.jsonl

# Recommended settings:
# - Models: Up to 70B with QLoRA, 8B with full fine-tuning
# - Memory: 24GB+ VRAM
# - Method: Full fine-tuning or LoRA
# - Expected time: 1-4 hours depending on method
```

#### **CPU-Only Systems**
```bash
# CPU training for testing (very slow)
uv run python cli.py finetune start --strategy cpu_small_model --dataset data.jsonl

# Recommended settings:
# - Models: 3B parameters maximum
# - Memory: 8GB+ RAM
# - Method: LoRA only
# - Expected time: 6-24 hours
```

#### **Multi-GPU/Cloud Setups**
```bash
# Distributed training for large models
uv run python cli.py finetune start --strategy cloud_scalable --dataset data.jsonl

# Recommended settings:
# - Models: 70B+ parameters
# - Hardware: 4+ GPUs with 40GB+ VRAM each
# - Method: QLoRA for efficiency
# - Expected time: 1-2 hours with proper setup
```

### 📊 **Resource Requirements Table**

| Model Size | Method | Min Memory | Recommended | Training Time | Suitable Hardware |
|------------|--------|------------|-------------|---------------|-------------------|
| **3B** | LoRA | 8GB | 16GB | 1-3 hours | Mac M1, RTX 3060+ |
| **8B** | LoRA | 12GB | 24GB | 2-4 hours | Mac Studio, RTX 4070+ |
| **8B** | Full | 32GB | 48GB | 4-8 hours | RTX 4090, A100 |
| **70B** | QLoRA | 24GB | 40GB | 2-6 hours | A100, H100 |
| **70B** | Full | 280GB | 400GB | 8-24 hours | 8x A100 |

### 📝 **Dataset Formats**

#### **Supported Formats**
- **JSONL**: JSON Lines format (recommended)
- **JSON**: Standard JSON arrays
- **Alpaca**: Instruction-input-output format
- **CSV**: Comma-separated values

#### **Alpaca Format Example**
```json
{
  "instruction": "Write a Python function to calculate factorial",
  "input": "",
  "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
}
```

#### **Conversation Format Example**
```json
{
  "text": "### Instruction:\nExplain machine learning\n\n### Response:\nMachine learning is a subset of AI that enables computers to learn from data..."
}
```

### 🔧 **Advanced Configuration**

#### **Configuration File Example**
```yaml
# fine_tuning_config.yaml
base_model:
  name: "llama3.1-8b"
  huggingface_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  torch_dtype: "bfloat16"

method:
  type: "lora"
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training_args:
  output_dir: "./fine_tuned_models"
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.03
  max_seq_length: 2048
  
dataset:
  path: "./training_data.jsonl"
  data_format: "jsonl"
  conversation_template: "alpaca"

environment:
  device: "auto"
  seed: 42
```

#### **Using Custom Configuration**
```bash
uv run python cli.py finetune start --config fine_tuning_config.yaml --dataset my_data.jsonl
```

### 📈 **Training Management**

#### **Monitor Training Progress**
```bash
# Start training and get job ID
uv run python cli.py finetune start --strategy mac_m1_lora --dataset data.jsonl

# Monitor progress (not yet implemented)
uv run python cli.py finetune monitor --job-id <job-id>

# List all training jobs (not yet implemented)
uv run python cli.py finetune jobs
```

#### **Resume from Checkpoint**
```bash
# Resume training from checkpoint (not yet implemented)
uv run python cli.py finetune resume --checkpoint ./fine_tuned_models/checkpoint-1000
```

### 🧪 **Model Evaluation & Export**

#### **Evaluate Fine-Tuned Model**
```bash
# Evaluate model performance (not yet implemented)
uv run python cli.py finetune evaluate --model-path ./fine_tuned_models/final

# Export model for deployment (not yet implemented)
uv run python cli.py finetune export --model-path ./fine_tuned_models/final --output ./exported_model
```

### 🎯 **Best Practices**

#### **Data Preparation**
1. **Quality over Quantity**: 1000 high-quality examples > 10000 poor examples
2. **Consistent Formatting**: Use consistent templates and formatting
3. **Balanced Dataset**: Ensure diverse examples across your use cases
4. **Data Validation**: Always validate your dataset format before training

#### **Training Tips**
1. **Start Small**: Begin with LoRA on a small model for testing
2. **Monitor Resources**: Use `--dry-run` to estimate requirements first
3. **Save Checkpoints**: Regular checkpointing prevents data loss
4. **Experiment Tracking**: Keep detailed logs of experiments and results

#### **Hardware Optimization**
1. **Mac Users**: Use strategy `mac_m1_lora` for optimized Metal Performance
2. **GPU Users**: Enable mixed precision (bf16) for faster training
3. **CPU Users**: Only for testing - consider cloud GPU for production
4. **Multi-GPU**: Use `cloud_scalable` strategy for distributed training

### 🔍 **Troubleshooting**

#### **Common Issues**

**Out of Memory (OOM)**
```bash
# Reduce batch size and increase gradient accumulation
# Or switch to QLoRA method
uv run python cli.py finetune start --strategy cpu_small_model  # Fallback to CPU
```

**Slow Training on Mac**
```bash
# Ensure Metal Performance Shaders are enabled
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python cli.py finetune start --strategy mac_m1_lora
```

**Model Not Found**
```bash
# Check available models and verify HuggingFace model ID
uv run python cli.py finetune strategies show mac_m1_lora
```

#### **Performance Optimization**
- **Batch Size**: Start with 1-2, increase gradually based on memory
- **Sequence Length**: Reduce max_seq_length if hitting memory limits
- **Gradient Accumulation**: Use to simulate larger batch sizes
- **Mixed Precision**: Enable bf16 on modern hardware for speed

### 📚 **Integration Examples**

#### **With RAG System**
```bash
# Generate training data from RAG evaluations
cd ../rag && uv run python cli.py search "topic" --output rag_context.json
cd ../models && uv run python cli.py finetune start --dataset rag_context.json
```

#### **With Prompt System**
```bash
# Use optimized prompts for training data generation
cd ../prompts && uv run python cli.py execute --template data_generation
cd ../models && uv run python cli.py finetune start --dataset generated_data.jsonl
```

## 🔗 Integration with LlamaFarm Ecosystem

### 🧠 **RAG System Integration**
```bash
# Use models with RAG context
cd ../rag && uv run python cli.py search "topic" | \
  cd ../models && uv run python cli.py query "Summarize this context" --provider openai_gpt4o_mini
```

### 📝 **Prompts System Integration**  
```bash
# Use optimized prompts with models
cd ../prompts && uv run python -m prompts.cli execute "query" --template medical_qa | \
  cd ../models && uv run python cli.py query - --provider anthropic_claude_3_haiku
```

### 🔄 **Unified Workflows**
- **RAG → Models**: Retrieved context + model generation
- **Prompts → Models**: Optimized prompts + model execution  
- **Models → Training**: Generated data + fine-tuning pipelines

## 🛠️ Troubleshooting

### ❗ **Critical Setup Issues**

#### **OpenAI Organization Header Error** ⚠️ 
```bash
# Error: "OpenAI-Organization header should match organization for API key"
# This is the most common issue - here's the fix:

# 1. Check your .env file and COMMENT OUT the OPENAI_ORG_ID line:
# OPENAI_ORG_ID=  # Optional: your organization ID (commented out to prevent header issues)

# 2. Restart your demo/CLI after making this change

# 3. Test that it's fixed:
uv run python cli.py test openai_gpt4o_mini
```

#### **Environment Configuration Setup**
```bash
# 1. Copy the environment template:
cp ../.env.example ../.env

# 2. Add your OpenAI API key to .env:
OPENAI_API_KEY=sk-your-key-here

# 3. IMPORTANT: Comment out or remove OPENAI_ORG_ID:
# OPENAI_ORG_ID=  # Optional: your organization ID (commented out to prevent header issues)

# 4. Test your configuration:
uv run python cli.py validate-config
```

### ❗ **Common Demo Issues**

#### **API Key Problems**
```bash
# Check environment variables
env | grep API_KEY

# Validate configuration
uv run python cli.py validate-config

# Test specific provider
uv run python cli.py test openai_gpt4o_mini

# If you get "quota exceeded" but have quota, check your API key is loaded:
echo $OPENAI_API_KEY  # Should show your key
```

#### **Demo Quick Tests**
```bash
# Test the fixed demos with automated mode:
DEMO_MODE=automated uv run python demos/demo_fallback.py
DEMO_MODE=automated uv run python demos/demo_multi_model.py
DEMO_MODE=automated uv run python demos/demo_pytorch.py --skip-training --skip-ollama
DEMO_MODE=automated uv run python demos/demo_llamafactory.py --validate-only
```

#### **Ollama Issues**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Pull missing models
uv run python cli.py pull llama3.1:8b
```

#### **Model Not Found**
```bash
# List available models
uv run python cli.py list
uv run python cli.py list-local

# Check configuration
uv run python cli.py --config your_config.json list
```

### 🔍 **Debugging Tools**
```bash
# Health check all providers
uv run python cli.py health-check

# Detailed provider information
uv run python cli.py list --detailed

# Test with verbose output
uv run python cli.py query "test" --provider openai_gpt4o_mini --json
```

## 🧪 Testing & Development

### **Run Tests**
```bash
# Unit tests (34 tests)
uv run python -m pytest tests/test_models.py -v

# Integration tests (requires API keys)
uv run python -m pytest tests/test_e2e.py -v

# Specific provider tests
uv run python -m pytest tests/test_models.py::TestOllamaIntegration -v
```

### **Development Setup**
```bash
# Install development dependencies
uv sync --dev

# Run with coverage
uv run python -m pytest --cov=. --cov-report=html

# Format code
uv run black cli.py
```

## 📚 Documentation & Resources

- **[Developer Structure Guide](STRUCTURE.md)** - Internal architecture and development patterns
- **[API Integration Examples](docs/WORKING_API_CALLS.md)** - Real API call demonstrations
- **[Feature Verification](docs/ALL_WORKING_CONFIRMED.md)** - Complete feature testing results
- **[Configuration Examples](examples/)** - Working configuration templates

## 🤝 Contributing

We welcome contributions! Please see:
1. **[STRUCTURE.md](STRUCTURE.md)** - Developer architecture guide
2. **[GitHub Issues](../../issues)** - Bug reports and feature requests  
3. **Test Requirements** - All new features must include tests
4. **Documentation** - Update relevant docs with changes

## 📊 Current Status

- ✅ **34/34 unit tests passing**
- ✅ **12/12 integration tests passing**  
- ✅ **25+ CLI commands working**
- ✅ **Real API integration confirmed**
- ✅ **Production-ready configurations**
- 🚧 **Training infrastructure in development**

---

## 🦙 Ready to wrangle some models? No prob-llama!

Get started with your AI model management journey using the LlamaFarm Models System! 🚀