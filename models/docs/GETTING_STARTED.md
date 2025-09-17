# Getting Started Guide

Quick start guide to get you up and running with LlamaFarm Models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Examples](#basic-examples)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Optional
- NVIDIA GPU for training (CUDA 11.8+)
- Apple Silicon Mac for MPS acceleration
- API keys for cloud providers

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm/models
```

### Step 2: Install Dependencies

Using uv (recommended):
```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv sync
```

Using pip:
```bash
pip install -e .
```

### Step 3: Set Up Environment

Create a `.env` file:
```bash
# Cloud API Keys (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
HF_TOKEN=hf_...

# Local Model Settings
OLLAMA_BASE_URL=http://localhost:11434
MODEL_CACHE_DIR=./model_cache
```

## Quick Start

### Option 1: Local Models (No API Key Required)

```bash
# 1. Install Ollama (if not installed)
# macOS/Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve

# 3. Pull a model
ollama pull llama3.2:3b

# 4. Test with LlamaFarm
uv run python cli.py generate --strategy local_development "Hello, world!"
```

### Option 2: Cloud APIs

```bash
# 1. Set your API key
export OPENAI_API_KEY="your-key-here"

# 2. Use cloud strategy
uv run python cli.py generate --strategy cloud_production "Write a haiku about AI"
```

### Option 3: Hybrid (Recommended)

```bash
# Uses cloud when available, falls back to local
uv run python cli.py generate --strategy hybrid_fallback "Explain quantum computing"
```

## Basic Examples

### 1. Text Generation

```bash
# Simple generation
uv run python cli.py generate "Tell me a joke"

# With parameters
uv run python cli.py generate \
  --max-tokens 200 \
  --temperature 0.7 \
  "Write a short story"
```

### 2. Interactive Chat

```bash
# Start chat session
uv run python cli.py chat

# With specific model
uv run python cli.py chat --model gpt-3.5-turbo

# With system prompt
uv run python cli.py chat --system "You are a helpful coding assistant"
```

### 3. List Available Strategies

```bash
# See all pre-configured strategies
uv run python cli.py list-strategies

# Get details about a strategy
uv run python cli.py info --strategy cloud_production
```

### 4. Compare Models

```bash
# Compare responses from different models
uv run python cli.py compare \
  --models "gpt-3.5-turbo,llama3.2:3b" \
  --prompt "What is machine learning?"
```

## Working with Strategies

### Understanding Strategies

Strategies are pre-configured setups for different use cases:

| Strategy | Best For | Requirements |
|----------|----------|--------------|
| `local_development` | Offline development | Ollama installed |
| `cloud_production` | Production apps | OpenAI API key |
| `hybrid_fallback` | Best of both | Optional API key |
| `cost_optimized` | Budget constraints | Any provider |
| `privacy_first` | Sensitive data | Local models only |

### Using a Strategy

```bash
# Method 1: Command-line flag
uv run python cli.py generate --strategy local_development "Your prompt"

# Method 2: Set as default
uv run python cli.py use-strategy local_development
uv run python cli.py generate "Your prompt"  # Uses local_development

# Method 3: Custom strategy file
uv run python cli.py --strategy-file my_strategies.yaml generate "Your prompt"
```

### Creating Your First Custom Strategy

Create `my_strategy.yaml`:

```yaml
version: "2.0.0"

strategies:
  - name: my_first_strategy
    description: "My custom configuration"
    
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          api_key: ${OPENAI_API_KEY}
          default_model: gpt-3.5-turbo
          max_tokens: 500
    
    constraints:
      max_cost_per_request: 0.01  # Limit costs
```

Use it:
```bash
uv run python cli.py --strategy-file my_strategy.yaml \
  generate --strategy my_first_strategy "Hello!"
```

## Python API Usage

### Basic Example

```python
from models.core import ModelManager

# Initialize with strategy
manager = ModelManager(strategy="local_development")

# Generate text
response = manager.generate("What is Python?")
print(response)
```

### Advanced Example

```python
from models.core import StrategyManager, ModelManager

# Load custom strategies
strategy_mgr = StrategyManager(strategies_file="my_strategies.yaml")
strategy = strategy_mgr.get_strategy("my_strategy")

# Use with model manager
manager = ModelManager(strategy=strategy)

# Generate with parameters
response = manager.generate(
    prompt="Write code to sort a list",
    max_tokens=200,
    temperature=0.5
)
print(response)
```

## Next Steps

### Explore More Features

1. **Fine-Tuning**: Train custom models
   ```bash
   # See Training Guide
   uv run python cli.py train --help
   ```

2. **Model Management**: Work with different models
   ```bash
   # Ollama models
   uv run python cli.py ollama list
   
   # HuggingFace models
   uv run python cli.py list-hf
   ```

3. **Batch Processing**: Process multiple prompts
   ```bash
   uv run python cli.py batch --input prompts.txt --output results.jsonl
   ```

### Read the Documentation

- [CLI Reference](CLI.md) - All available commands
- [Strategies Guide](STRATEGIES.md) - Deep dive into strategies
- [Training Guide](TRAINING.md) - Fine-tuning and training
- [Architecture Guide](ARCHITECTURE.md) - How it all works

### Join the Community

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Contributing: PRs welcome!

## Troubleshooting

### Common Issues

#### "No API key found"
```bash
# Solution: Set your API key
export OPENAI_API_KEY="your-key"

# Or use local models
uv run python cli.py generate --strategy local_development "test"
```

#### "Ollama not running"
```bash
# Solution: Start Ollama
ollama serve

# Or use cloud strategy
uv run python cli.py generate --strategy cloud_production "test"
```

#### "Model not found"
```bash
# For Ollama:
ollama pull llama3.2:3b

# For cloud: Check model name
uv run python cli.py catalog list --provider openai
```

#### "Out of memory"
```bash
# Use smaller model or quantization
# See Training Guide for memory optimization
```

### Getting Help

```bash
# General help
uv run python cli.py --help

# Command-specific help
uv run python cli.py generate --help

# Check system status
uv run python cli.py health-check
```

### Debug Mode

```bash
# Enable debug logging
uv run python cli.py --log-level DEBUG generate "test"

# Verbose output
uv run python cli.py --verbose generate "test"
```

## Tips for Success

1. **Start Local**: Begin with `local_development` strategy
2. **Add Cloud Gradually**: Try `hybrid_fallback` when ready
3. **Monitor Costs**: Use constraints in production
4. **Cache Responses**: Enable caching for repeated queries
5. **Test Fallbacks**: Ensure your fallback chain works

## Example Workflows

### Development Workflow
```bash
# 1. Setup local environment
uv run python cli.py setup demos/strategies.yaml --strategy local_development

# 2. Test generation
uv run python cli.py generate --strategy local_development "test"

# 3. Start coding
uv run python cli.py chat --strategy local_development
```

### Production Workflow
```bash
# 1. Validate configuration
uv run python cli.py validate-config production_strategies.yaml

# 2. Test all providers
uv run python cli.py health-check

# 3. Deploy with monitoring
uv run python cli.py generate --strategy cloud_production --verbose "test"
```

### Experimentation Workflow
```bash
# 1. Compare models
uv run python cli.py compare --models "gpt-4o,gpt-3.5-turbo,llama3.2:3b" "test"

# 2. Test different parameters
uv run python cli.py generate --temperature 0.1 "precise answer"
uv run python cli.py generate --temperature 0.9 "creative story"

# 3. Measure performance
time uv run python cli.py generate --strategy fast_inference "quick test"
```

Ready to build something amazing? Start with the examples above and explore the full documentation for advanced features!