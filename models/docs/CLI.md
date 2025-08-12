# 🖥️ LlamaFarm CLI Documentation

> **Complete Command-Line Interface Guide** - Strategy-based model management, setup automation, and training

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Command Reference](#-command-reference)
- [Strategies System](#-strategies-system)
- [Setup & Installation](#-setup--installation)
- [Model Operations](#-model-operations)
- [Training & Fine-tuning](#-training--fine-tuning)
- [Mock Models & Testing](#-mock-models--testing)
- [Examples](#-examples)

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/llamafarm.git
cd llamafarm/models

# Install dependencies
uv sync

# Setup environment
cp ../.env.example ../.env
# Edit .env with your API keys

# Run automatic setup
uv run python cli.py setup demos/strategies.yaml --auto
```

### First Commands
```bash
# List available strategies
uv run python cli.py list-strategies

# Get info about a strategy
uv run python cli.py info --strategy mock_development

# Use mock model for testing (no API key needed)
uv run python cli.py info --strategy mock_development

# Run a demo
DEMO_MODE=automated uv run python demos/demo1_cloud_fallback.py
```

## 🎯 Core Concepts

### Strategies
Strategies are pre-configured model setups that define:
- Which models/APIs to use
- Fallback chains for reliability
- Hardware optimization settings
- Training configurations
- Export/conversion options

### Components
Reusable building blocks that strategies use:
- **model_app**: Model servers (Ollama, mock)
- **cloud_api**: Cloud providers (OpenAI, Anthropic)
- **fine_tuner**: Training engines (PyTorch, LlamaFactory)
- **converter**: Format converters (GGUF, Ollama)

### Setup Manager
Automatic installation and configuration system that:
- Analyzes strategy requirements
- Installs needed components
- Downloads required models
- Configures environment

## 📚 Command Reference

### Setup Commands

#### `setup` - Install Required Components
```bash
# Verify requirements only
uv run python cli.py setup <strategy_file> --verify-only

# Automatic installation
uv run python cli.py setup <strategy_file> --auto

# Interactive mode (default)
uv run python cli.py setup <strategy_file>

# Verbose output
uv run python cli.py setup <strategy_file> --verbose
```

**Examples:**
```bash
# Setup for mock development
uv run python cli.py setup demos/mock_strategy.yaml --auto

# Setup for training
uv run python cli.py setup demos/strategies.yaml --auto

# Check requirements without installing
uv run python cli.py setup demos/strategies.yaml --verify-only
```

### Strategy Commands

#### `list-strategies` - List Available Strategies
```bash
uv run python cli.py list-strategies
```

Output shows all strategies in the default strategies.yaml file.

#### `info` - Get Strategy Information
```bash
# Get info about specific strategy
uv run python cli.py info --strategy <strategy_name>

# Export strategy configuration
uv run python cli.py info --strategy <strategy_name> --export
```

**Examples:**
```bash
# Info about mock development strategy
uv run python cli.py info --strategy mock_development

# Info about training strategy
uv run python cli.py info --strategy demo3_training
```

#### `use-strategy` - Set Active Strategy
```bash
uv run python cli.py use-strategy <strategy_name>
```

### Model Operations

#### `list` - List Available Models
```bash
# List all configured models
uv run python cli.py list

# Detailed view with costs
uv run python cli.py list --detailed
```

#### `test` - Test Model Connection
```bash
# Test specific provider
uv run python cli.py test <provider_name>

# Test all providers
uv run python cli.py health-check
```

#### `query` - Send Single Query
```bash
# Basic query
uv run python cli.py query "Your question here"

# With specific provider
uv run python cli.py query "Question" --provider <provider_name>

# With parameters
uv run python cli.py query "Question" --temperature 0.8 --max-tokens 500
```

#### `chat` - Interactive Chat Session
```bash
# Start chat with default model
uv run python cli.py chat

# With specific provider
uv run python cli.py chat --provider <provider_name>

# With system prompt
uv run python cli.py chat --system "You are a helpful assistant"

# Save/load history
uv run python cli.py chat --save-history session.json
uv run python cli.py chat --history previous_session.json
```

### Local Model Management

#### `list-local` - List Ollama Models
```bash
uv run python cli.py list-local
```

#### `pull` - Download Ollama Model
```bash
# Pull specific model
uv run python cli.py pull llama3.2:3b
uv run python cli.py pull mistral:latest
```

#### `test-local` - Test Local Model
```bash
uv run python cli.py test-local llama3.1:8b
```

### Training Commands

#### `train` - Start Training
```bash
# With strategy
uv run python cli.py train --strategy <strategy_name> --dataset <data.jsonl>

# Custom parameters
uv run python cli.py train \
  --dataset data.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 \
  --batch-size 4
```

#### `finetune` - Advanced Fine-tuning
```bash
# List fine-tuning strategies
uv run python cli.py finetune strategies list

# Start with strategy
uv run python cli.py finetune start --strategy mac_m1_lora --dataset data.jsonl

# Estimate resources
uv run python cli.py finetune estimate --strategy gpu_full_finetune
```

#### `convert` - Convert Model Formats
```bash
# Convert to GGUF
uv run python cli.py convert --input model_path --output model.gguf --format gguf

# Convert to Ollama
uv run python cli.py convert --input model.gguf --output ollama_model --format ollama
```

## 🎨 Strategies System

### Default Strategies

#### Development Strategies
- **mock_development** - Mock model for testing (no API needed)
- **local_development** - Local Ollama models
- **demo1_cloud_fallback** - Cloud with automatic fallback
- **demo2_multi_model** - Multiple model configuration

#### Training Strategies
- **demo3_training** - Basic PyTorch training
- **training_cuda_consumer** - Consumer GPU optimization
- **training_mps_apple** - Apple Silicon optimization
- **training_cpu_only** - CPU-only training
- **training_cuda_datacenter** - Multi-GPU setup

#### Production Strategies
- **production_hybrid** - Cloud + local fallback
- **local_development** - Privacy-first local models

### Creating Custom Strategies

Create a YAML file with your strategy:

```yaml
version: "2.0"
strategies:
  my_custom_strategy:
    name: "My Custom Setup"
    description: "Custom configuration for my use case"
    
    components:
      model_app:
        type: ollama
        config:
          base_url: "http://localhost:11434"
          default_model: "llama3.2:3b"
      
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          default_model: "gpt-4o-mini"
    
    fallback_chain:
      - cloud_api
      - model_app
    
    routing_rules:
      - condition:
          prompt_contains: ["code", "programming"]
        action:
          use_component: cloud_api
```

## 🔧 Setup & Installation

### Component Installation

The setup system automatically installs required components:

#### Supported Components
- **ollama** - Local model server
- **gguf_converter** - GGUF format converter (uses llama.cpp)
- **mock_model** - Built-in mock for testing
- **pytorch** - PyTorch training framework
- **llamafactory** - Advanced training system

#### Installation Methods
- **builtin** - No installation needed
- **homebrew** - macOS package manager
- **script** - Shell script execution
- **download** - Direct download
- **build_from_source** - Compile from source

### Platform Support

#### macOS (Apple Silicon)
```bash
# Optimized for M1/M2/M3
uv run python cli.py setup demos/strategies.yaml --auto

# Uses Metal Performance Shaders for training
uv run python cli.py train --strategy training_mps_apple --dataset data.jsonl
```

#### Linux/Windows
```bash
# CUDA support for NVIDIA GPUs
uv run python cli.py setup demos/strategies.yaml --auto

# Uses CUDA for training
uv run python cli.py train --strategy training_cuda_consumer --dataset data.jsonl
```

## 🧪 Mock Models & Testing

### Using Mock Models

Mock models are perfect for:
- Testing without API keys
- Development and debugging
- Unit testing
- CI/CD pipelines
- Quick prototyping

```bash
# Setup mock model (instant, no download)
uv run python cli.py setup demos/mock_strategy.yaml --auto

# Test mock model
uv run python cli.py info --strategy mock_development

# Run mock demo
DEMO_MODE=automated uv run python demos/demo_mock_model.py
```

### Available Mock Models
- **mock-gpt-4** - Simulates GPT-4 responses
- **mock-claude-3** - Simulates Claude responses
- **mock-tiny** - Lightweight mock model

## 📋 Examples

### Example 1: Quick Start with Mock
```bash
# No API keys needed!
uv run python cli.py setup demos/mock_strategy.yaml --auto
uv run python cli.py info --strategy mock_development
```

### Example 2: Setup Cloud + Fallback
```bash
# Setup OpenAI with Ollama fallback
uv run python cli.py setup demos/strategies.yaml --auto

# Test the setup
uv run python cli.py info --strategy demo1_cloud_fallback
```

### Example 3: Train a Model
```bash
# Setup training environment
uv run python cli.py setup demos/strategies.yaml --auto

# Start training
uv run python cli.py train \
  --strategy demo3_training \
  --dataset demos/datasets/medical/medical_qa.jsonl \
  --epochs 1 \
  --batch-size 2
```

### Example 4: Convert Model to Ollama
```bash
# Train model
uv run python cli.py train --strategy demo3_training --dataset data.jsonl

# Convert to GGUF
uv run python cli.py convert \
  --input ./fine_tuned_models/final_model \
  --output model.gguf \
  --format gguf

# Create Ollama model
ollama create my_model -f Modelfile
```

## 🐛 Troubleshooting

### Common Issues

#### "Strategy not found"
```bash
# Check available strategies
uv run python cli.py list-strategies

# Verify strategy file exists
ls demos/strategies.yaml
```

#### "Component not installed"
```bash
# Run setup to install components
uv run python cli.py setup demos/strategies.yaml --auto

# Verify installation
uv run python cli.py setup demos/strategies.yaml --verify-only
```

#### "API key not set"
```bash
# Check environment
cat ../.env | grep API_KEY

# Set in .env file
echo "OPENAI_API_KEY=sk-..." >> ../.env
```

#### "Ollama not running"
```bash
# Start Ollama
ollama serve

# Check status
curl http://localhost:11434/api/tags
```

### Debug Mode
```bash
# Run with verbose output
uv run python cli.py --log-level DEBUG <command>

# Check configuration
uv run python cli.py validate-config
```

## 🧪 Testing

### Run Tests
```bash
# Run CLI tests
uv run pytest tests/test_cli_mock.py -v

# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_cli_mock.py::TestCLIMockIntegration -v
```

### Test Coverage
- ✅ CLI command parsing
- ✅ Strategy management
- ✅ Setup automation
- ✅ Mock model integration
- ✅ Error handling
- ✅ End-to-end workflows

## 📚 Additional Resources

- [Main README](../README.md) - Project overview
- [Strategies Documentation](../demos/strategies.yaml) - Strategy configurations
- [Component Definitions](../components/definitions/) - Component specifications
- [Demo Scripts](../demos/) - Example implementations

## 🤝 Contributing

Contributions welcome! Please:
1. Add tests for new features
2. Update documentation
3. Follow existing patterns
4. Run tests before submitting

---

**Ready to manage models like a pro? The CLI is your command center! 🚀**