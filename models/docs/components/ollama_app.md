# Ollama Model App

Ollama provides a simple way to run large language models locally with an easy-to-use API.

## Features

- **Local Model Execution**: Run models entirely on your machine
- **Simple API**: REST API compatible with OpenAI format
- **Model Management**: Easy model downloading and management
- **GPU Acceleration**: Automatic GPU detection and usage
- **Streaming Support**: Real-time response streaming
- **Custom Models**: Create custom models with Modelfiles

## Installation

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

## Configuration

### Basic Configuration

```yaml
type: "ollama"
config:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 300
  auto_start: true
```

### Advanced Configuration

```yaml
type: "ollama"
config:
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
  timeout: 300
  auto_start: true
  gpu_layers: 35  # Number of layers to offload to GPU
  num_thread: 8   # CPU threads to use
  models:
    - name: "llama3.2"
      pull_on_start: true
    - name: "mistral"
      pull_on_start: false
```

## Available Models

Ollama supports many models out of the box:

- **LLaMA 3.2**: 1B, 3B versions
- **LLaMA 3.1**: 8B, 70B, 405B versions
- **Mistral**: 7B model
- **CodeLlama**: Various sizes for code generation
- **Phi-3**: Microsoft's small models
- **Gemma**: Google's open models
- **Qwen**: Alibaba's models

## Usage Examples

### Basic Usage

```python
from models.components.model_apps.ollama import OllamaApp

# Initialize
config = {
    "base_url": "http://localhost:11434",
    "default_model": "llama3.2"
}
ollama = OllamaApp(config)

# Start service if needed
if not ollama.is_running():
    ollama.start_service()

# Generate text
response = ollama.generate("Explain quantum computing in simple terms")
print(response)

# Stream responses
for chunk in ollama.generate("Write a story about AI", stream=True):
    print(chunk, end="", flush=True)
```

### Chat Interface

```python
# Chat conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is machine learning?"}
]

response = ollama.chat(messages)
print(response)

# Streaming chat
for chunk in ollama.chat(messages, stream=True):
    print(chunk, end="", flush=True)
```

### Model Management

```python
# List available models
models = ollama.list_models()
for model in models:
    print(f"{model['name']}: {model['size']}")

# Pull a new model
success = ollama.pull_model("mistral")

# Get model information
info = ollama.get_model_info("llama3.2")
print(f"Model: {info['name']}")
print(f"Parameters: {info['parameters']}")

# Delete a model
ollama.delete_model("old-model")
```

### Custom Models

Create custom models with Modelfiles:

```python
modelfile = """
FROM llama3.2

# Set temperature
PARAMETER temperature 0.7

# Set system message
SYSTEM You are a helpful coding assistant specializing in Python.
"""

# Create custom model
ollama.create_model("python-assistant", modelfile)

# Use custom model
response = ollama.generate(
    "Write a Python function to sort a list",
    model="python-assistant"
)
```

## Generation Parameters

```python
response = ollama.generate(
    "Your prompt here",
    temperature=0.7,      # Creativity (0-2)
    max_tokens=500,       # Maximum response length
    top_p=0.9,           # Nucleus sampling
    top_k=40,            # Top-k sampling
    seed=42              # For reproducibility
)
```

## Performance Tips

1. **GPU Usage**: Ollama automatically uses GPU if available
2. **Model Selection**: Choose smaller models for faster responses
3. **Quantization**: Use quantized models (Q4, Q5) for better performance
4. **Context Length**: Limit context for faster generation

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   ```bash
   # Check if port is in use
   lsof -i :11434
   
   # Start manually
   ollama serve
   ```

2. **GPU Not Detected**
   ```bash
   # Check GPU support
   ollama run llama3.2 --verbose
   ```

3. **Model Download Issues**
   ```bash
   # Manual download
   ollama pull llama3.2
   
   # Check models directory
   ls ~/.ollama/models
   ```

## Integration with Other Components

Ollama can be used with:
- **RAG Systems**: As an embedder or generator
- **Fine-tuning**: Export fine-tuned models to Ollama format
- **Agents**: As the LLM backend for agent systems

## Resource Requirements

| Model | RAM Required | Disk Space |
|-------|--------------|------------|
| LLaMA 3.2 1B | 2GB | 1.3GB |
| LLaMA 3.2 3B | 4GB | 2GB |
| LLaMA 3.1 8B | 8GB | 4.7GB |
| Mistral 7B | 8GB | 4.1GB |
| CodeLlama 13B | 16GB | 7.4GB |

## Security Considerations

- Ollama runs locally by default (localhost only)
- To expose externally: `OLLAMA_HOST=0.0.0.0 ollama serve`
- Use firewall rules to restrict access
- No data leaves your machine