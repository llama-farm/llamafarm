# CLI Reference

Complete reference for the LlamaFarm Models CLI commands.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Strategy Commands](#strategy-commands)
- [Generation Commands](#generation-commands)
- [Training Commands](#training-commands)
- [Model Management](#model-management)
- [Utilities](#utilities)

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Basic Usage

```bash
# General format
uv run python cli.py [command] [options]

# Get help
uv run python cli.py --help

# Get help for specific command
uv run python cli.py [command] --help
```

## Strategy Commands

Strategies are the core configuration system in LlamaFarm.

### list-strategies
List all available strategies from the default configuration.

```bash
uv run python cli.py list-strategies

# With custom strategy file
uv run python cli.py --strategy-file custom_strategies.yaml list-strategies
```

### use-strategy
Switch to and use a specific strategy.

```bash
uv run python cli.py use-strategy local_development
uv run python cli.py use-strategy cloud_production
```

### info
Show detailed information about a strategy.

```bash
# Current strategy
uv run python cli.py info

# Specific strategy
uv run python cli.py info --strategy cloud_production
```

### setup
Setup tools and models required by a strategy.

```bash
# Auto-setup everything
uv run python cli.py setup demos/strategies.yaml --auto

# Verify without installing
uv run python cli.py setup demos/strategies.yaml --verify-only

# Setup specific strategy
uv run python cli.py setup demos/strategies.yaml --strategy local_development
```

## Generation Commands

### generate
Generate text using the current strategy.

```bash
# Basic generation
uv run python cli.py generate "Write a poem about AI"

# With specific strategy
uv run python cli.py generate --strategy local_development "Hello world"

# With parameters
uv run python cli.py generate \
  --max-tokens 500 \
  --temperature 0.7 \
  "Your prompt"
```

### complete
Get text completion (provider-agnostic).

```bash
uv run python cli.py complete \
  --prompt "The future of AI is" \
  --max-tokens 100 \
  --strategy cloud_production
```

### chat
Start an interactive chat session.

```bash
# Default strategy
uv run python cli.py chat

# With specific model
uv run python cli.py chat --model gpt-4o-mini

# With system prompt
uv run python cli.py chat --system "You are a helpful assistant"
```

### query
Send a query with full control over parameters.

```bash
uv run python cli.py query \
  --provider openai \
  --model gpt-3.5-turbo \
  --temperature 0.5 \
  "Your question"
```

## Training Commands

### train
Train a model using a strategy.

```bash
# Basic training
uv run python cli.py train \
  --strategy demo3_training \
  --dataset data.jsonl \
  --epochs 3

# With validation
uv run python cli.py train \
  --strategy fine_tuning_pipeline \
  --dataset train.jsonl \
  --eval-dataset eval.jsonl \
  --epochs 5 \
  --batch-size 4
```

### finetune
Fine-tuning operations with detailed control.

```bash
# Start fine-tuning
uv run python cli.py finetune start \
  --model "meta-llama/Llama-2-7b-hf" \
  --dataset train.jsonl \
  --method lora \
  --output ./output

# Monitor progress
uv run python cli.py finetune monitor --job-id [job_id]

# List jobs
uv run python cli.py finetune jobs

# Evaluate model
uv run python cli.py finetune evaluate \
  --model ./output/checkpoint-best \
  --dataset test.jsonl
```

### datasplit
Split dataset into train/eval sets.

```bash
uv run python cli.py datasplit \
  --input data.jsonl \
  --train-output train.jsonl \
  --eval-output eval.jsonl \
  --eval-percentage 10 \
  --seed 42
```

## Model Management

### Ollama Commands

```bash
# List installed models
uv run python cli.py ollama list

# Pull a model
uv run python cli.py ollama pull llama3.2:3b

# Run a model
uv run python cli.py ollama run llama3.2:3b "Your prompt"

# Check status
uv run python cli.py ollama status
```

### HuggingFace Commands

```bash
# List available models
uv run python cli.py list-hf --category text-generation

# Download a model
uv run python cli.py download-hf meta-llama/Llama-2-7b-hf

# Test a model
uv run python cli.py test-hf \
  --model meta-llama/Llama-2-7b-hf \
  --prompt "Test prompt"

# Login to HuggingFace
uv run python cli.py hf-login
```

### Model Catalog

```bash
# List models from catalog
uv run python cli.py catalog list
uv run python cli.py catalog list --provider openai
uv run python cli.py catalog list --category vision

# Search models
uv run python cli.py catalog search "code generation"

# Show model info
uv run python cli.py catalog info gpt-4o

# Show fallback chains
uv run python cli.py catalog fallbacks
```

## Utilities

### convert
Convert models between formats.

```bash
# Convert to GGUF format
uv run python cli.py convert \
  --input model_path \
  --output model.gguf \
  --format gguf

# Convert to Ollama format
uv run python cli.py convert \
  --input model.gguf \
  --output ollama_model \
  --format ollama \
  --quantization q4_k_m
```

### test
Test model connectivity.

```bash
# Test current strategy
uv run python cli.py test

# Test specific provider
uv run python cli.py test --provider openai

# Test local models
uv run python cli.py test-local
```

### validate-config
Validate configuration files.

```bash
uv run python cli.py validate-config custom_strategies.yaml
```

### compare
Compare responses from different models.

```bash
uv run python cli.py compare \
  --models "gpt-3.5-turbo,gpt-4o-mini,llama3.2:3b" \
  --prompt "Explain quantum computing"
```

### batch
Process multiple queries from a file.

```bash
# Process queries
uv run python cli.py batch \
  --input queries.txt \
  --output responses.jsonl \
  --provider openai \
  --model gpt-3.5-turbo
```

### health-check
Check health of all configured providers.

```bash
uv run python cli.py health-check
```

## Configuration

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
export HF_TOKEN="hf_..."

# Endpoints
export OLLAMA_BASE_URL="http://localhost:11434"

# Directories
export MODEL_CACHE_DIR="./model_cache"
export OUTPUT_DIR="./output"
```

### Strategy Files

Strategies are defined in YAML files:

```yaml
version: "2.0"
strategies:
  - name: my_custom_strategy
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
```

### Command-Line Options

Global options available for all commands:

```bash
--strategy-file PATH    # Path to strategy YAML file
--strategy NAME         # Strategy to use
--log-level LEVEL      # Logging level: DEBUG, INFO, WARNING, ERROR
--output-format FORMAT # Output format: text, json, yaml
--verbose             # Enable verbose output
--quiet              # Suppress non-essential output
```

## Examples

### Common Workflows

#### 1. Local Development Setup
```bash
# Setup local Ollama
uv run python cli.py setup demos/strategies.yaml --strategy local_development --auto

# Test it works
uv run python cli.py generate --strategy local_development "Hello"

# Start interactive chat
uv run python cli.py chat --strategy local_development
```

#### 2. Cloud API Usage
```bash
# Setup environment
export OPENAI_API_KEY="sk-..."

# Use cloud strategy
uv run python cli.py generate --strategy cloud_production "Write code"

# Compare models
uv run python cli.py compare \
  --models "gpt-3.5-turbo,gpt-4o" \
  --prompt "Complex question"
```

#### 3. Fine-Tuning Workflow
```bash
# Prepare data
uv run python cli.py datasplit \
  --input data.jsonl \
  --eval-percentage 10

# Start training
uv run python cli.py train \
  --strategy fine_tuning_pipeline \
  --dataset data_train.jsonl \
  --eval-dataset data_eval.jsonl \
  --epochs 3

# Test the model
uv run python cli.py generate \
  --model ./output/checkpoint-best \
  --prompt "Test prompt"
```

## Error Handling

Common errors and solutions:

```bash
# Missing API key
export OPENAI_API_KEY="your-key"

# Ollama not running
ollama serve  # In another terminal

# Model not found
uv run python cli.py ollama pull model-name

# Out of memory
# Reduce batch size or use quantization
```

## Tips

1. **Use strategies** instead of manual configuration
2. **Set up environment variables** in `.env` file
3. **Test connectivity** before long operations
4. **Monitor training** with TensorBoard
5. **Save checkpoints** regularly during training