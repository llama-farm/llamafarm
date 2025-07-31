# 🦙 LlamaFarm Models System

A comprehensive model management system for cloud and local LLMs, providing unified access to OpenAI, Anthropic, Ollama, Hugging Face, and more.

## Features

- **25+ CLI Commands** for complete model lifecycle management
- **Multi-Provider Support**: OpenAI, Anthropic, Together, Groq, Cohere, Ollama, Hugging Face
- **Real API Integration**: Makes actual API calls and returns real model responses
- **Local Model Integration**: Ollama, vLLM, Text Generation Inference (TGI)
- **Advanced Query Control**: Temperature, max tokens, system prompts, streaming
- **Interactive Features**: Chat sessions, batch processing, file sending
- **Fallback Chains**: Automatic failover between providers
- **Cost Tracking**: Monitor API usage and costs
- **Performance Monitoring**: Track latency and throughput

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment variables
cp ../.env.example ../.env
# Add your API keys to ../.env

# List available providers (uses default config)
uv run python cli.py list

# Send a query with default provider (OpenAI GPT-4o-mini)
uv run python cli.py query "What is machine learning?"

# Use a specific provider
uv run python cli.py query "Explain quantum computing" --provider openai_gpt4_turbo

# Start interactive chat
uv run python cli.py chat

# Use a different configuration file
uv run python cli.py --config config/real_models_example.json list
```

## ✅ Real API Responses

The Models system makes **actual API calls** and returns **real model responses**!

### OpenAI Example
```bash
$ uv run python cli.py query "What is machine learning?"
ℹ  Using provider: openai_gpt4o_mini
ℹ  Model: gpt-4o-mini  
✓ Response received in 4447ms

Sure! Machine learning is a type of technology that allows computers to learn from data and improve their performance over time without being explicitly programmed for each specific task.

Here's a simple way to think about it:

1. **Learning from Examples**: Just like how people learn by looking at examples, machines can learn by analyzing large amounts of data...
[Full detailed response continues]
```

### Ollama Local Model Example
```bash  
$ uv run python cli.py query "Tell me a joke" --provider ollama_llama3
ℹ  Using provider: ollama_llama3
ℹ  Model: llama3.1:8b
✓ Response received in 810ms

Here's one:

What do you call a fake noodle?

An impasta.
```

### System Prompt Example
```bash
$ uv run python cli.py query "Explain quantum computing" --system "You are a physics professor"
ℹ  Using provider: openai_gpt4o_mini
ℹ  Model: gpt-4o-mini
✓ Response received in 17592ms

Quantum computing is an advanced computational paradigm that leverages the principles of quantum mechanics...
[Full technical response continues]
```

### Code Review Example
```bash
$ uv run python cli.py send code.py --prompt "Review this code" --provider ollama_llama3
ℹ  Sending file to: ollama_llama3
ℹ  Model: llama3.1:8b
✓ Response received in 6582ms

**Code Review**

The provided code calculates the Fibonacci sequence. However, it has a few issues:

1. **Inefficient Recursion**: The current implementation uses recursive function calls...
[Full detailed code review continues]
```

## Complete CLI Reference

### Core Model Interaction Commands

#### `query` - Send queries with full parameter control
```bash
# Basic query
uv run python cli.py query "Explain quantum computing"

# Use specific provider
uv run python cli.py query "Write a Python function" --provider openai_gpt4o_mini

# Override temperature and max tokens
uv run python cli.py query "Generate creative story" --temperature 0.9 --max-tokens 500

# Add system prompt
uv run python cli.py query "Analyze this data: [1,2,3,4,5]" --system "You are a data scientist"

# Stream response
uv run python cli.py query "Tell me a long story" --stream

# Output as JSON
uv run python cli.py query "List 5 facts about AI" --json

# Save response to file
uv run python cli.py query "Write a README template" --save output.md
```

#### `chat` - Interactive chat sessions
```bash
# Start basic chat
uv run python cli.py chat

# Chat with specific provider
uv run python cli.py chat --provider anthropic_claude_3_haiku

# Set system prompt for chat
uv run python cli.py chat --system "You are a helpful coding assistant"

# Load and continue previous chat
uv run python cli.py chat --history previous_chat.json

# Save chat history
uv run python cli.py chat --save-history my_chat.json

# Adjust temperature for creativity
uv run python cli.py chat --temperature 0.8
```

#### `send` - Send file contents to models
```bash
# Send a code file for review
uv run python cli.py send code.py --prompt "Review this code for bugs"

# Send with specific provider
uv run python cli.py send document.txt --provider openai_gpt4_turbo

# Save analysis to file
uv run python cli.py send data.csv --prompt "Analyze this data" --output analysis.md

# Control generation parameters
uv run python cli.py send script.js --temperature 0.2 --max-tokens 1000
```

#### `batch` - Process multiple queries
```bash
# Process queries from file (one per line)
uv run python cli.py batch queries.txt

# Use specific provider for batch
uv run python cli.py batch prompts.txt --provider openai_gpt4o_mini

# Save all responses
uv run python cli.py batch questions.txt --output responses.json

# Process with parallel requests
uv run python cli.py batch large_batch.txt --parallel 5

# Set temperature for all queries
uv run python cli.py batch creative_prompts.txt --temperature 0.9
```

### Testing and Management Commands

#### `test` - Test provider connectivity
```bash
# Test specific provider
uv run python cli.py test openai_gpt4o_mini

# Test with custom query
uv run python cli.py test anthropic_claude_3_haiku --query "Hello, Claude!"
```

#### `compare` - Compare responses from multiple models
```bash
# Compare two models
uv run python cli.py compare --providers openai_gpt4o_mini,anthropic_claude_3_haiku --query "Explain recursion"

# Compare multiple models
uv run python cli.py compare --providers openai_gpt4_turbo,anthropic_claude_3_opus,together_llama3_70b --query "Write a sorting algorithm"
```

#### `list` - List configured providers
```bash
# Basic listing
uv run python cli.py list

# Detailed view with costs and settings
uv run python cli.py list --detailed
```

#### `health-check` - Check all providers
```bash
# Run health check on all providers
uv run python cli.py health-check
```

#### `validate-config` - Validate configuration
```bash
# Validate default config
uv run python cli.py validate-config

# Validate specific config file
uv run python cli.py --config custom_config.json validate-config
```

### Configuration Generation Commands

#### `generate-config` - Generate configuration templates
```bash
# Generate basic config
uv run python cli.py generate-config --type basic

# Generate multi-provider config with fallbacks
uv run python cli.py generate-config --type multi --output multi_provider.json

# Generate production-ready config
uv run python cli.py generate-config --type production --output prod_config.json
```

### Ollama Integration Commands

#### `list-local` - List Ollama models
```bash
# List all local Ollama models
uv run python cli.py list-local
```

#### `pull` - Download Ollama models
```bash
# Pull a specific model
uv run python cli.py pull llama3.2:3b

# Pull latest version
uv run python cli.py pull mistral:latest
```

#### `test-local` - Test Ollama models
```bash
# Test with default query
uv run python cli.py test-local llama3.1:8b

# Test with custom query
uv run python cli.py test-local codellama:13b --query "Write a Python function to sort a list"
```

#### `generate-ollama-config` - Generate Ollama configuration
```bash
# Generate config with all local models
uv run python cli.py generate-ollama-config

# Save to specific file
uv run python cli.py generate-ollama-config --output ollama_models.json
```

### Hugging Face Integration Commands

#### `hf-login` - Login to Hugging Face Hub
```bash
# Login with token from environment
uv run python cli.py hf-login
```

#### `list-hf` - Search Hugging Face models
```bash
# Search for models
uv run python cli.py list-hf --search "gpt2"

# Limit results
uv run python cli.py list-hf --search "llama" --limit 10
```

#### `download-hf` - Download models from Hub
```bash
# Download a model
uv run python cli.py download-hf gpt2

# Download to custom directory
uv run python cli.py download-hf distilbert-base-uncased --cache-dir ./models

# Include all files
uv run python cli.py download-hf bert-base-uncased --include-images
```

#### `test-hf` - Test Hugging Face models
```bash
# Test a model
uv run python cli.py test-hf gpt2 --query "Once upon a time"

# Set max tokens
uv run python cli.py test-hf distilgpt2 --query "Hello" --max-tokens 50

# Use GPU if available
uv run python cli.py test-hf gpt2-medium --query "Test" --gpu
```

#### `generate-hf-config` - Generate HF configuration
```bash
# Generate default config
uv run python cli.py generate-hf-config

# Include specific models
uv run python cli.py generate-hf-config --models "gpt2,distilgpt2,gpt2-medium"

# Save to file
uv run python cli.py generate-hf-config --output hf_config.json
```

### Local Inference Engine Commands

#### `list-vllm` - List vLLM compatible models
```bash
# List popular vLLM models
uv run python cli.py list-vllm
```

#### `test-vllm` - Test models with vLLM
```bash
# Test a model
uv run python cli.py test-vllm meta-llama/Llama-2-7b-chat-hf --query "Hello"

# Configure generation
uv run python cli.py test-vllm mistralai/Mistral-7B-v0.1 --query "Test" --max-tokens 100

# Set GPU memory usage
uv run python cli.py test-vllm model_name --query "Test" --gpu-memory 0.8
```

#### `list-tgi` - List TGI endpoints
```bash
# List configured TGI endpoints
uv run python cli.py list-tgi
```

#### `test-tgi` - Test TGI endpoints
```bash
# Test an endpoint
uv run python cli.py test-tgi --endpoint http://localhost:8080 --query "Hello"

# Set generation parameters
uv run python cli.py test-tgi --endpoint http://tgi.local --query "Test" --max-tokens 50
```

#### `generate-engines-config` - Generate local engines config
```bash
# Generate config for available engines
uv run python cli.py generate-engines-config

# Include unavailable engines as examples
uv run python cli.py generate-engines-config --include-unavailable

# Save to file
uv run python cli.py generate-engines-config --output engines.json
```

## Configuration Examples

### Basic Configuration
```json
{
  "name": "My Models Configuration",
  "version": "1.0.0",
  "default_provider": "openai_gpt4o_mini",
  "providers": {
    "openai_gpt4o_mini": {
      "type": "cloud",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key": "${OPENAI_API_KEY}",
      "temperature": 0.7,
      "max_tokens": 2048
    }
  }
}
```

### Multi-Provider with Fallback
```json
{
  "name": "Production Configuration",
  "version": "1.0.0",
  "default_provider": "primary",
  "fallback_chain": ["primary", "secondary", "local_backup"],
  "providers": {
    "primary": {
      "type": "cloud",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "api_key": "${OPENAI_API_KEY}"
    },
    "secondary": {
      "type": "cloud",
      "provider": "anthropic",
      "model": "claude-3-haiku-20240307",
      "api_key": "${ANTHROPIC_API_KEY}"
    },
    "local_backup": {
      "type": "local",
      "provider": "ollama",
      "model": "llama3.1:8b",
      "host": "localhost"
    }
  }
}
```

### Use Case Specific Configurations

See `config/use_case_examples.json` for configurations optimized for:
- RAG Systems with embeddings
- Code generation and review
- Customer support chatbots
- Content generation (creative, factual, SEO)
- Data analysis
- Translation services
- Educational assistants

### Real Model Examples

See `config/real_models_example.json` for a comprehensive list of:
- **OpenAI**: GPT-4 Turbo, GPT-4o-mini, GPT-3.5 Turbo
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Together AI**: Llama 3.1 70B, Mixtral 8x7B
- **Groq**: Ultra-fast Llama 3 70B, Mixtral
- **Cohere**: Command R+
- **Ollama**: Llama 3.1/3.2, Mistral, Phi-3, CodeLlama
- **Hugging Face**: GPT-2, DistilGPT-2
- **vLLM**: High-performance local inference
- **TGI**: Text Generation Inference endpoints

## Environment Variables

Create a `.env` file in the parent directory:

```bash
# Cloud Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...

# Hugging Face
HF_TOKEN=hf_...

# Local Models
OLLAMA_HOST=localhost  # Optional, defaults to localhost
```

## Advanced Usage

### Using Custom Configurations
```bash
# Use a custom config file for all commands
uv run python cli.py --config my_config.json list
uv run python cli.py --config prod_config.json query "Hello"
```

### Streaming Responses
```bash
# Stream responses in real-time
uv run python cli.py query "Tell me a story" --stream
uv run python cli.py chat --provider openai_gpt4o_mini  # Chat always supports streaming
```

### Cost-Aware Usage
```bash
# Use cost-effective models for simple tasks
uv run python cli.py query "What is 2+2?" --provider openai_gpt4o_mini

# Use powerful models for complex tasks
uv run python cli.py query "Explain quantum entanglement in detail" --provider openai_gpt4_turbo
```

### Performance Optimization
```bash
# Use Groq for ultra-fast responses
uv run python cli.py query "Quick question" --provider groq_llama3_70b

# Use local models to avoid network latency
uv run python cli.py query "Test query" --provider ollama_phi3
```

### Batch Processing for Efficiency
```bash
# Process many queries efficiently
echo "Question 1\nQuestion 2\nQuestion 3" > queries.txt
uv run python cli.py batch queries.txt --parallel 3 --output results.json
```

## Integration with LlamaFarm RAG

The Models system integrates seamlessly with the RAG system:

```bash
# Use RAG to find context, then query with models
cd ../rag && uv run python cli.py search "llama care" | \
  cd ../models && uv run python cli.py send - --prompt "Summarize this information"

# Use specific model for RAG responses
cd ../models && uv run python cli.py query \
  "Based on the following context: [RAG_CONTEXT], answer: What is llama grooming?" \
  --provider openai_gpt4o_mini \
  --temperature 0.3
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   # Check your environment
   env | grep API_KEY
   
   # Validate configuration
   uv run python cli.py validate-config
   ```

2. **Ollama Connection Issues**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama if needed
   ollama serve
   ```

3. **Model Not Found**
   ```bash
   # List available models
   uv run python cli.py list
   uv run python cli.py list-local
   
   # Pull missing Ollama model
   uv run python cli.py pull llama3.1:8b
   ```

4. **Timeout Issues**
   ```bash
   # Increase timeout in config
   "providers": {
     "slow_model": {
       "timeout": 300  // 5 minutes
     }
   }
   ```

## Testing

```bash
# Run unit tests
uv run python -m pytest tests/test_models.py -v

# Run integration tests (requires API keys)
uv run python -m pytest tests/test_e2e.py -v

# Run specific test
uv run python -m pytest tests/test_models.py::TestOllamaIntegration -v
```

Current test results: **34/34 unit tests passing** ✅

## Development

### Running Tests
```bash
# Run unit tests
uv run python -m pytest tests/test_models.py -v

# Run integration tests (requires API keys)
uv run python -m pytest tests/test_e2e.py -v

# Run specific test
uv run python -m pytest tests/test_models.py::TestOllamaIntegration -v
```

### Adding New Providers

1. Add provider configuration schema
2. Implement API client in `cli.py`
3. Add tests in `tests/`
4. Update documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Directory Structure

```
models/
├── cli.py                     # Main CLI application
├── README.md                  # This documentation
├── pyproject.toml            # Python project configuration
├── setup_and_demo.sh         # Setup and demo script
├── config/                   # Configuration files
│   ├── default.json          # Default configuration (auto-loaded)
│   ├── real_models_example.json  # Comprehensive real model configs
│   ├── use_case_examples.json    # Use-case specific configurations
│   └── test_config.json      # Test configuration
├── docs/                     # Documentation
│   ├── ALL_WORKING_CONFIRMED.md   # API integration confirmation
│   ├── ENHANCEMENTS.md       # Recent enhancements overview
│   ├── FIXED_CONFIGURATION.md     # Configuration fixes
│   └── WORKING_API_CALLS.md  # Real API call examples
├── examples/                 # Example configurations and demos
│   ├── demo_*.json          # Generated demo configurations
│   └── config_examples/     # Additional config examples
└── tests/                   # Test suites
    ├── test_models.py       # Unit tests (34 tests)
    └── test_e2e.py         # Integration tests (12 tests)
```

## Additional Documentation

- **[Working API Calls](docs/WORKING_API_CALLS.md)** - Real API integration examples
- **[All Features Confirmed](docs/ALL_WORKING_CONFIRMED.md)** - Complete feature verification
- **[Recent Enhancements](docs/ENHANCEMENTS.md)** - Latest improvements and additions
- **[Configuration Fixes](docs/FIXED_CONFIGURATION.md)** - Configuration system improvements

## 🦙 No prob-llama with model management!