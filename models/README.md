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

## 🎓 Training & Fine-Tuning Roadmap

> **Coming Soon**: Comprehensive training workflows for custom model development

### 🚀 **Phase 1: Data Preparation (Q4 2024)**

#### **Dataset Management**
```bash
# Planned commands for data preparation
uv run python cli.py data prepare --source raw_data/ --output processed/
uv run python cli.py data validate --dataset processed/ --format jsonl
uv run python cli.py data split --dataset processed/ --train 0.8 --val 0.1 --test 0.1
```

**Features:**
- **Automated data cleaning** and format conversion
- **Quality filtering** with configurable thresholds  
- **Train/validation/test** splitting with stratification
- **Data deduplication** and privacy filtering
- **Format conversion** (JSON, JSONL, Parquet, CSV)

#### **Domain-Specific Datasets**
- **RAG Training Data**: Question-context-answer triplets
- **Code Generation**: Repository analysis and function documentation
- **Conversation Data**: Multi-turn dialogue formatting
- **Classification Data**: Label balancing and augmentation

### 🔬 **Phase 2: Fine-Tuning Infrastructure (Q1 2025)**

#### **Local Fine-Tuning Support**
```bash
# Planned fine-tuning commands
uv run python cli.py train start --model llama3.1:8b --dataset my_data.jsonl
uv run python cli.py train monitor --job job_12345
uv run python cli.py train evaluate --model fine_tuned/checkpoint-1000
```

**Supported Methods:**
- **LoRA/QLoRA**: Parameter-efficient fine-tuning
- **Full Fine-Tuning**: Complete model retraining
- **Instruction Tuning**: Chat and instruction following
- **RLHF**: Reinforcement learning from human feedback

#### **Cloud Fine-Tuning Integration**
- **OpenAI Fine-Tuning**: GPT-3.5/4 custom models
- **Together AI**: Llama and Mistral fine-tuning
- **Hugging Face AutoTrain**: Automated training pipelines
- **Custom Training**: Integration with training platforms

### 🎯 **Phase 3: Advanced Training (Q2 2025)**

#### **Multi-Modal Training**
```bash
# Planned multi-modal support
uv run python cli.py train vision --model llava --dataset image_text_pairs/
uv run python cli.py train audio --model whisper --dataset speech_transcripts/
```

#### **Specialized Training Workflows**
- **RAG Model Training**: Retrieval-augmented generation optimization
- **Code Model Training**: Programming language specialization  
- **Domain Adaptation**: Medical, legal, financial model variants
- **Multilingual Training**: Cross-language transfer learning

#### **Training Infrastructure**
- **Distributed Training**: Multi-GPU and multi-node support
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Model Versioning**: Automated model registry and deployment
- **Performance Monitoring**: Training metrics and evaluation pipelines

### 🏭 **Phase 4: Production Deployment (Q3 2025)**

#### **Model Serving**
```bash
# Planned deployment commands
uv run python cli.py deploy start --model my_fine_tuned_model --port 8080
uv run python cli.py deploy scale --replicas 3 --gpu-per-replica 1
uv run python cli.py deploy monitor --metrics latency,throughput,accuracy
```

#### **Production Features**
- **Auto-scaling**: Dynamic resource allocation based on load
- **A/B Testing**: Model variant comparison in production
- **Performance Monitoring**: Real-time metrics and alerting
- **Cost Optimization**: Efficient resource utilization

### 📚 **Training Resources & Documentation**

#### **Getting Started Guides** (Available Now)
- **[Training Best Practices](docs/TRAINING_BEST_PRACTICES.md)** - Coming Soon
- **[Data Preparation Guide](docs/DATA_PREPARATION.md)** - Coming Soon  
- **[Fine-Tuning Cookbook](docs/FINE_TUNING_COOKBOOK.md)** - Coming Soon
- **[Production Deployment](docs/PRODUCTION_DEPLOYMENT.md)** - Coming Soon

#### **Training Templates** (Planned)
- **Instruction Following**: Format and optimize for chat models
- **Code Generation**: Programming task specialization
- **RAG Optimization**: Retrieval-augmented generation improvement
- **Domain Specialization**: Medical, legal, financial model variants

#### **Integration Examples**
```python
# Planned Python API for training workflows
from llamafarm_models.training import FineTuningPipeline

# Initialize training pipeline
trainer = FineTuningPipeline(
    base_model="llama3.1:8b",
    dataset="my_training_data.jsonl", 
    method="lora",
    output_dir="./fine_tuned_models"
)

# Start training
trainer.train(
    epochs=3,
    learning_rate=1e-4,
    batch_size=16
)

# Evaluate results
metrics = trainer.evaluate()
print(f"Training completed: {metrics}")
```

### 🛠️ **Current Training Capabilities**

While full training infrastructure is in development, you can currently:

#### **Prepare for Training**
```bash
# Generate training data from conversations
uv run python cli.py chat --save-history training_conversations.json

# Process files for training data
uv run python cli.py send documents/ --prompt "Generate Q&A pairs" --output training_data.txt

# Batch process for dataset creation
uv run python cli.py batch question_prompts.txt --output answers.jsonl
```

#### **Model Evaluation**
```bash
# Compare models for training baseline
uv run python cli.py compare --providers openai_gpt4o_mini,ollama_llama3 --query "Evaluate this task"

# Test custom prompts across models
uv run python cli.py batch evaluation_questions.txt --provider your_model
```

#### **Integration Planning**
The models system is designed to integrate seamlessly with:
- **LlamaFarm RAG**: Training data from RAG evaluations
- **LlamaFarm Prompts**: Prompt optimization and evaluation
- **Custom Training Pipelines**: API-compatible model serving

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

### ❗ **Common Issues**

#### **API Key Problems**
```bash
# Check environment variables
env | grep API_KEY

# Validate configuration
uv run python cli.py validate-config

# Test specific provider
uv run python cli.py test openai_gpt4o_mini
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