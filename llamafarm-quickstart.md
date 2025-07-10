# 🚀 LlamaFarm Quick Start Guide

Get your first AI model running locally in under 5 minutes!

## Prerequisites

- Node.js 18+ installed
- 8GB+ of RAM recommended
- ~10GB free disk space for models

## Installation

```bash
# Install globally via npm
npm install -g @llamafarm/llamafarm

# Or clone and build from source
git clone https://github.com/llamafarm/llamafarm-cli.git
cd llamafarm-cli
npm install
npm run build
npm link
```

## Your First Harvest 🌾

### 1. Initialize Your Farm

```bash
# Set up your farm configuration
llamafarm till

# This will ask you about:
# - Your preferred device (Mac, Linux, Windows, Pi, etc.)
# - Default model settings
# - Vector database preferences
# - Agent framework choice
```

### 2. Plant a Simple Chat Model

```bash
# Plant Llama 3 8B with a basic chat interface
llamafarm plant llama3-8b --agent chat-basic

# What this does:
# 🌱 Downloads the model (if needed)
# 🤖 Configures a chat agent
# 🌐 Sets up a web UI
# 📦 Packages everything into a binary
# 🔗 Provides a download link
```

### 3. Harvest Your Model

```bash
# Download and deploy the packaged binary
llamafarm harvest http://localhost:8080/download/v3.1/llamafarm-llama3-8b-mac.tar.gz

# This extracts and sets up:
# - The model files
# - Agent configuration
# - Web interface
# - Startup scripts
```

### 4. Run Your Harvest

```bash
cd harvests
./run-llamafarm-llama3-8b.sh

# Your AI is now running at http://localhost:8080
```

## Advanced Example: RAG-Powered Assistant 📚

```bash
# 1. Set up vector database
llamafarm silo --init chroma

# 2. Add your documents
llamafarm sow ./documents --type pdf --chunk-size 512

# 3. Plant with RAG enabled
llamafarm plant llama3-8b \
  --agent research-assistant \
  --rag enabled \
  --database vector \
  --config my-rag-config.yaml

# 4. Harvest and run
llamafarm harvest http://localhost:8080/download/v3.1/...
```

## Using YAML Configuration

Create `farm.yaml`:

```yaml
model:
  name: mixtral-8x7b
  quantization: q4_0

agent:
  name: coding-assistant
  framework: langchain
  tools:
    - code_interpreter
    - web_search
  
database:
  type: vector
  provider: chroma

rag:
  enabled: true
  chunk_size: 512

deployment:
  device: mac
  gpu: true
  port: 8080
```

Then plant with:
```bash
llamafarm plant mixtral-8x7b --config farm.yaml
```

## Testing Your Models 🧪

```bash
# Run benchmarks
llamafarm greenhouse --model llama3-8b --benchmark

# Test with scenarios
llamafarm greenhouse --scenario tests/customer-service.json
```

## Useful Commands

```bash
# Check system status
llamafarm weather

# View available recipes
llamafarm almanac --list

# Clean up old deployments
llamafarm compost --days 30

# List stored models
llamafarm barn --list
```

## Troubleshooting

### Model won't download?
```bash
# Check your internet connection
# Ensure you have enough disk space
# Try a smaller model first (phi-2, tinyllama)
```

### Port already in use?
```bash
# LlamaFarm auto-assigns ports, but you can specify:
llamafarm plant llama3-8b --port 3000
```

### GPU not detected?
```bash
# Check GPU support
llamafarm weather --detailed

# Disable GPU if needed
llamafarm plant llama3-8b --gpu false
```

## Next Steps

1. **Explore the Almanac**: `llamafarm almanac` for recipes and patterns
2. **Join the Community**: Share your harvests and get help
3. **Contribute**: Add new features, models, or agents

Happy farming! 🦙🌾

---

**Need help?** Open an issue on [GitHub](https://github.com/llamafarm/llamafarm-cli/issues)