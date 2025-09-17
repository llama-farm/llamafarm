# OpenAI-Compatible API

The OpenAI-Compatible API component provides access to multiple AI providers using the OpenAI SDK interface, making it easy to switch between providers without code changes.

## Supported Providers

- **OpenAI** - GPT models (GPT-4, GPT-3.5, etc.)
- **Together.ai** - Open-source models (Llama, Mixtral, etc.)
- **Anyscale** - Scalable inference
- **Perplexity** - Internet-connected models
- **Groq** - Ultra-fast LPU inference
- **Mistral AI** - European AI models
- **DeepSeek** - Cost-effective models
- **xAI (Grok)** - X/Twitter integrated models
- **Fireworks AI** - Fast inference platform
- **OpenRouter** - Multi-provider gateway
- **Custom** - Any OpenAI-compatible endpoint

See [COMPATIBLE_MODELS.md](./COMPATIBLE_MODELS.md) for detailed provider information.

## Features

- **Universal Interface**: Same API for all providers
- **Chat & Completion**: Both chat and completion endpoints
- **Streaming**: Real-time response streaming
- **Embeddings**: Text embedding generation (provider-dependent)
- **Moderation**: Content moderation (OpenAI only)
- **Token Counting**: Accurate token counting with tiktoken
- **Function Calling**: Support for function/tool calling
- **Automatic Fallback**: Easy provider switching

## Configuration

### Basic Configuration

```yaml
# OpenAI
type: "openai_compatible"
config:
  provider: "openai"
  api_key: "${OPENAI_API_KEY}"  # Can use environment variable
  default_model: "gpt-4o-mini"

# Together.ai
type: "openai_compatible"
config:
  provider: "together"
  api_key: "${TOGETHER_API_KEY}"
  default_model: "meta-llama/Llama-3-70b-chat-hf"

# Groq (Fast Inference)
type: "openai_compatible"
config:
  provider: "groq"
  api_key: "${GROQ_API_KEY}"
  default_model: "llama-3.1-70b-versatile"
```

### Custom Endpoint Configuration

```yaml
type: "openai_compatible"
config:
  provider: "custom"
  base_url: "https://your-api-endpoint.com/v1"
  api_key: "${CUSTOM_API_KEY}"
  default_model: "your-model-name"
```

## Usage Examples

### Basic Chat (Any Provider)

```python
from models.components.cloud_apis.openai_compatible import OpenAICompatibleAPI

# Initialize with OpenAI
config = {
    "provider": "openai",
    "api_key": "your-api-key",
    "default_model": "gpt-4o-mini"
}
api = OpenAICompatibleAPI(config)

# Or initialize with Groq for faster inference
config = {
    "provider": "groq",
    "api_key": "your-groq-key",
    "default_model": "llama-3.1-70b-versatile"
}
api = OpenAICompatibleAPI(config)

# Same interface for all providers
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain quantum computing"}
]

response = api.chat(messages)
print(response)
```

### Provider Switching

```python
# Easy provider switching without code changes
providers = [
    {"provider": "openai", "model": "gpt-3.5-turbo"},
    {"provider": "groq", "model": "llama-3.1-8b-instant"},
    {"provider": "together", "model": "meta-llama/Llama-3-70b-chat-hf"}
]

for provider_config in providers:
    config = {
        "provider": provider_config["provider"],
        "api_key": os.getenv(f"{provider_config['provider'].upper()}_API_KEY"),
        "default_model": provider_config["model"]
    }
    
    api = OpenAICompatibleAPI(config)
    response = api.chat(messages)
    print(f"{provider_config['provider']}: {response[:100]}...")
```

### Streaming Responses

```python
# Stream for real-time output (works with all providers)
for chunk in api.chat(messages, stream=True):
    print(chunk, end="", flush=True)
```

### Advanced Parameters

```python
response = api.chat(
    messages,
    model="gpt-4",  # Or any model from your provider
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n\n", "END"]
)
```

### Using Perplexity for Internet Search

```python
# Perplexity with internet access
config = {
    "provider": "perplexity",
    "api_key": os.getenv("PERPLEXITY_API_KEY"),
    "default_model": "llama-3.1-sonar-large-128k-online"
}
api = OpenAICompatibleAPI(config)

messages = [
    {"role": "user", "content": "What are the latest developments in quantum computing as of 2024?"}
]
response = api.chat(messages)
print(response)  # Will include current information from the internet
```

### Cost Optimization with Provider Selection

```python
def select_provider_by_task(task_type):
    """Select optimal provider based on task requirements"""
    
    if task_type == "simple":
        # Use fast, cheap provider for simple tasks
        return {
            "provider": "groq",
            "model": "llama-3.1-8b-instant"
        }
    elif task_type == "complex":
        # Use powerful model for complex reasoning
        return {
            "provider": "openai",
            "model": "gpt-4o"
        }
    elif task_type == "code":
        # Use specialized code model
        return {
            "provider": "deepseek",
            "model": "deepseek-coder"
        }
    elif task_type == "research":
        # Use internet-connected model
        return {
            "provider": "perplexity",
            "model": "llama-3.1-sonar-large-128k-online"
        }
```

### Embeddings

```python
# Create embeddings (provider-dependent)
text = "Machine learning is fascinating"

# OpenAI embeddings
openai_api = OpenAICompatibleAPI({"provider": "openai", "api_key": key})
embedding = openai_api.create_embedding(text, model="text-embedding-ada-002")

# Mistral embeddings
mistral_api = OpenAICompatibleAPI({"provider": "mistral", "api_key": key})
embedding = mistral_api.create_embedding(text, model="mistral-embed")
```

### Error Handling with Fallback

```python
def chat_with_fallback(messages, providers):
    """Try multiple providers with automatic fallback"""
    
    for provider_name in providers:
        try:
            config = {
                "provider": provider_name,
                "api_key": os.getenv(f"{provider_name.upper()}_API_KEY")
            }
            api = OpenAICompatibleAPI(config)
            
            # Validate credentials
            if api.validate_credentials():
                return api.chat(messages)
                
        except Exception as e:
            print(f"Provider {provider_name} failed: {e}")
            continue
    
    raise Exception("All providers failed")

# Use with fallback chain
response = chat_with_fallback(
    messages,
    ["groq", "together", "openai"]  # Try Groq first, then Together, then OpenAI
)
```

## Best Practices

### 1. API Key Security
```python
# Use environment variables for each provider
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["TOGETHER_API_KEY"] = "..."
os.environ["GROQ_API_KEY"] = "gsk_..."

# Load from environment
config = {
    "provider": "groq",
    "api_key": os.getenv("GROQ_API_KEY")
}
```

### 2. Provider Selection Strategy
- **Development**: Use cheaper/faster providers (Groq, Together)
- **Production**: Use reliable providers (OpenAI, Mistral)
- **Cost-sensitive**: Use DeepSeek, Together
- **Speed-critical**: Use Groq (LPU technology)
- **Research tasks**: Use Perplexity (internet access)

### 3. Model Selection
```python
MODEL_SELECTION = {
    "fast": {
        "groq": "llama-3.1-8b-instant",
        "together": "meta-llama/Llama-3-8b-chat-hf"
    },
    "balanced": {
        "openai": "gpt-3.5-turbo",
        "mistral": "mistral-small-latest"
    },
    "powerful": {
        "openai": "gpt-4o",
        "mistral": "mistral-large-latest"
    }
}
```

### 4. Rate Limit Handling
```python
import time
from typing import Optional

def chat_with_retry(api, messages, max_retries=3):
    """Handle rate limits with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            return api.chat(messages)
        except Exception as e:
            if "rate" in str(e).lower():
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    
    raise Exception("Max retries exceeded")
```

## Provider Comparison

| Provider | Speed | Cost | Quality | Special Features |
|----------|-------|------|---------|------------------|
| OpenAI | Medium | High | Excellent | Most features, tools |
| Groq | Ultra-fast | Low | Good | LPU technology |
| Together | Fast | Low | Good | Many open models |
| Perplexity | Medium | Medium | Good | Internet access |
| Mistral | Fast | Medium | Very Good | European, GDPR |
| DeepSeek | Medium | Very Low | Good | Code specialist |
| Anyscale | Fast | Medium | Good | Scalable |
| Fireworks | Fast | Low | Good | Serverless |
| OpenRouter | Varies | Varies | Varies | Multi-provider |

## Migration Guide

### From OpenAI to OpenAI-Compatible

```python
# Old code (OpenAI only)
from models.components.cloud_apis.openai import OpenAIAPI
api = OpenAIAPI({"api_key": key})

# New code (Any provider)
from models.components.cloud_apis.openai_compatible import OpenAICompatibleAPI
api = OpenAICompatibleAPI({
    "provider": "groq",  # or "openai", "together", etc.
    "api_key": key
})
```

### Environment Variables

```bash
# Set up multiple providers
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export TOGETHER_API_KEY="..."
export MISTRAL_API_KEY="..."
export PERPLEXITY_API_KEY="pplx-..."
```

## Troubleshooting

### Common Issues

1. **Provider Not Found**
   - Check provider name spelling
   - Verify provider is in PROVIDER_CONFIGS

2. **API Key Issues**
   - Check environment variable name
   - Verify key format for provider

3. **Model Not Available**
   - Check provider's model list
   - Use provider's default model

4. **Rate Limiting**
   - Implement retry logic
   - Use multiple provider fallback

5. **Endpoint Errors**
   - Verify base_url for custom providers
   - Check provider service status

## Integration Tips

- **With RAG**: Use fast providers for retrieval, powerful for generation
- **With Fine-tuning**: Compare base vs fine-tuned across providers
- **With Agents**: Use OpenAI for tools, others for reasoning
- **With Evaluation**: Use multiple providers for consensus