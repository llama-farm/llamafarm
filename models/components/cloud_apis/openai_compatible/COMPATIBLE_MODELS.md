# OpenAI-Compatible API Providers

This document lists cloud-based AI models that use the OpenAI SDK/API template, making them easily interchangeable in your applications.

## Supported Providers

### 1. OpenAI
- **Endpoint**: `https://api.openai.com/v1` (default)
- **API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Environment Variable**: `OPENAI_API_KEY`
- **Popular Models**:
  - `gpt-4o` - Most capable model
  - `gpt-4o-mini` - Faster, cheaper GPT-4 variant
  - `gpt-4-turbo` - GPT-4 with vision capabilities
  - `gpt-3.5-turbo` - Fast and cost-effective
- **Documentation**: [OpenAI API Docs](https://platform.openai.com/docs)

### 2. Together.ai
- **Endpoint**: `https://api.together.xyz/v1`
- **API Key**: Get from [Together AI](https://api.together.xyz/)
- **Environment Variable**: `TOGETHER_API_KEY`
- **Popular Models**:
  - `meta-llama/Llama-3-70b-chat-hf`
  - `meta-llama/Llama-3-8b-chat-hf`
  - `mistralai/Mixtral-8x7B-Instruct-v0.1`
  - `NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO`
- **Documentation**: [Together AI Docs](https://docs.together.ai/)

### 3. Anyscale
- **Endpoint**: `https://api.endpoints.anyscale.com/v1`
- **API Key**: Get from [Anyscale Console](https://console.anyscale.com/)
- **Environment Variable**: `ANYSCALE_API_KEY`
- **Popular Models**:
  - `meta-llama/Llama-2-70b-chat-hf`
  - `meta-llama/Llama-2-13b-chat-hf`
  - `mistralai/Mistral-7B-Instruct-v0.1`
- **Documentation**: [Anyscale Endpoints](https://docs.anyscale.com/endpoints/)

### 4. Perplexity AI
- **Endpoint**: `https://api.perplexity.ai`
- **API Key**: Get from [Perplexity Settings](https://www.perplexity.ai/settings/api)
- **Environment Variable**: `PERPLEXITY_API_KEY`
- **Popular Models**:
  - `llama-3.1-sonar-large-128k-online` - With internet access
  - `llama-3.1-sonar-small-128k-online` - Smaller, with internet
  - `llama-3.1-sonar-large-128k-chat` - Without internet
  - `llama-3.1-sonar-small-128k-chat` - Smaller, without internet
- **Documentation**: [Perplexity API Docs](https://docs.perplexity.ai/)

### 5. Groq
- **Endpoint**: `https://api.groq.com/openai/v1`
- **API Key**: Get from [Groq Console](https://console.groq.com/)
- **Environment Variable**: `GROQ_API_KEY`
- **Popular Models**:
  - `llama-3.1-70b-versatile`
  - `llama-3.1-8b-instant`
  - `mixtral-8x7b-32768`
  - `gemma2-9b-it`
- **Special Feature**: Extremely fast inference (LPU technology)
- **Documentation**: [Groq Docs](https://console.groq.com/docs)

### 6. Mistral AI
- **Endpoint**: `https://api.mistral.ai/v1`
- **API Key**: Get from [Mistral Console](https://console.mistral.ai/)
- **Environment Variable**: `MISTRAL_API_KEY`
- **Popular Models**:
  - `mistral-large-latest` - Most capable
  - `mistral-medium-latest` - Balanced performance
  - `mistral-small-latest` - Fast and efficient
  - `codestral-latest` - Code generation
- **Documentation**: [Mistral AI Docs](https://docs.mistral.ai/)

### 7. DeepSeek
- **Endpoint**: `https://api.deepseek.com/v1`
- **API Key**: Get from [DeepSeek Platform](https://platform.deepseek.com/)
- **Environment Variable**: `DEEPSEEK_API_KEY`
- **Popular Models**:
  - `deepseek-chat` - General purpose chat
  - `deepseek-coder` - Code generation specialist
- **Documentation**: [DeepSeek API Docs](https://platform.deepseek.com/api-docs)

### 8. xAI (Grok)
- **Endpoint**: `https://api.x.ai/v1`
- **API Key**: Get from [xAI Console](https://console.x.ai/)
- **Environment Variable**: `XAI_API_KEY`
- **Popular Models**:
  - `grok-beta` - Latest Grok model
- **Documentation**: [xAI API Docs](https://docs.x.ai/)

### 9. Fireworks AI
- **Endpoint**: `https://api.fireworks.ai/inference/v1`
- **API Key**: Get from [Fireworks App](https://app.fireworks.ai/)
- **Environment Variable**: `FIREWORKS_API_KEY`
- **Popular Models**:
  - `accounts/fireworks/models/llama-v3p1-70b-instruct`
  - `accounts/fireworks/models/llama-v3p1-8b-instruct`
  - `accounts/fireworks/models/mixtral-8x7b-instruct`
- **Documentation**: [Fireworks Docs](https://readme.fireworks.ai/)

### 10. OpenRouter
- **Endpoint**: `https://openrouter.ai/api/v1`
- **API Key**: Get from [OpenRouter](https://openrouter.ai/)
- **Environment Variable**: `OPENROUTER_API_KEY`
- **Special Feature**: Access to multiple providers through one API
- **Popular Models**:
  - `meta-llama/llama-3.1-70b-instruct`
  - `anthropic/claude-3-opus`
  - `google/gemini-pro`
  - `openai/gpt-4`
- **Documentation**: [OpenRouter Docs](https://openrouter.ai/docs)

## Usage Example

```python
from models.components.cloud_apis.openai_compatible import OpenAICompatibleAPI

# Example 1: Using OpenAI
openai_config = {
    "provider": "openai",
    "api_key": "your-openai-api-key",  # or set OPENAI_API_KEY env var
    "default_model": "gpt-4o-mini"
}
openai_client = OpenAICompatibleAPI(openai_config)

# Example 2: Using Together.ai
together_config = {
    "provider": "together",
    "api_key": "your-together-api-key",  # or set TOGETHER_API_KEY env var
    "default_model": "meta-llama/Llama-3-70b-chat-hf"
}
together_client = OpenAICompatibleAPI(together_config)

# Example 3: Using Groq for fast inference
groq_config = {
    "provider": "groq",
    "api_key": "your-groq-api-key",  # or set GROQ_API_KEY env var
    "default_model": "llama-3.1-70b-versatile"
}
groq_client = OpenAICompatibleAPI(groq_config)

# Example 4: Custom endpoint
custom_config = {
    "provider": "custom",
    "base_url": "https://your-custom-endpoint.com/v1",
    "api_key": "your-api-key",
    "default_model": "your-model-name"
}
custom_client = OpenAICompatibleAPI(custom_config)

# All clients use the same interface
response = client.chat([{"role": "user", "content": "Hello!"}])
print(response)
```

## Choosing a Provider

### For General Purpose:
- **OpenAI**: Best overall quality, most features
- **Mistral**: Good balance of quality and cost
- **Together.ai**: Wide model selection, competitive pricing

### For Speed:
- **Groq**: Fastest inference (LPU technology)
- **Fireworks**: Fast inference, good pricing
- **Anyscale**: Optimized infrastructure

### For Cost-Effectiveness:
- **DeepSeek**: Very competitive pricing
- **Together.ai**: Good pricing for open models
- **OpenRouter**: Pay-as-you-go for multiple providers

### For Specialized Tasks:
- **Perplexity**: Internet-connected models for current information
- **DeepSeek Coder**: Specialized for code generation
- **Mistral Codestral**: Code generation with good performance
- **xAI Grok**: Access to real-time X (Twitter) data

### For Privacy/Compliance:
- **Anyscale**: Can deploy on your infrastructure
- **Together.ai**: Offers private deployments
- **Mistral**: European company, GDPR compliant

## Environment Setup

```bash
# Set up environment variables for the providers you want to use
export OPENAI_API_KEY="sk-..."
export TOGETHER_API_KEY="..."
export GROQ_API_KEY="gsk_..."
export MISTRAL_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export XAI_API_KEY="..."
export PERPLEXITY_API_KEY="pplx-..."
export ANYSCALE_API_KEY="..."
export FIREWORKS_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Rate Limits and Pricing

Each provider has different rate limits and pricing:

1. **OpenAI**: Tiered rate limits based on usage, pricing per 1K tokens
2. **Together.ai**: Generous rate limits, competitive open-model pricing
3. **Groq**: Free tier available, very fast but with rate limits
4. **Mistral**: Competitive European pricing, good rate limits
5. **Perplexity**: Limited requests in free tier, paid plans available
6. **DeepSeek**: Very competitive pricing, especially for code models
7. **Anyscale**: Enterprise-focused pricing
8. **Fireworks**: Usage-based pricing, good for burst workloads
9. **OpenRouter**: Pay-as-you-go, pricing varies by model
10. **xAI**: Premium pricing for access to Grok models

Always check the provider's current pricing page for the most up-to-date information.

## Migration Guide

To migrate from OpenAI to another provider:

1. Change the `provider` field in your configuration
2. Update the `default_model` to one supported by the new provider
3. Set the appropriate environment variable for the API key
4. No code changes needed - the API interface remains the same!

```python
# Before (OpenAI)
config = {
    "provider": "openai",
    "default_model": "gpt-3.5-turbo"
}

# After (Groq for faster inference)
config = {
    "provider": "groq",
    "default_model": "llama-3.1-8b-instant"
}
```

The OpenAICompatibleAPI class handles all the provider-specific details automatically.