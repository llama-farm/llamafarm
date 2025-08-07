# OpenAI API

The OpenAI API component provides access to GPT models and other OpenAI services.

## Features

- **GPT Models**: Access to GPT-3.5, GPT-4, and other models
- **Chat & Completion**: Both chat and completion endpoints
- **Streaming**: Real-time response streaming
- **Embeddings**: Text embedding generation
- **Moderation**: Content moderation API
- **Token Counting**: Accurate token counting with tiktoken
- **Function Calling**: Support for function/tool calling

## Configuration

### Basic Configuration

```yaml
type: "openai"
config:
  api_key: "${OPENAI_API_KEY}"  # Can use environment variable
  default_model: "gpt-3.5-turbo"
```

### Advanced Configuration

```yaml
type: "openai"
config:
  api_key: "${OPENAI_API_KEY}"
  organization: "org-xxxxx"  # Optional organization ID
  default_model: "gpt-4"
  timeout: 60
  max_retries: 3
  base_url: "https://api.openai.com/v1"  # For proxies/custom endpoints
```

## Available Models

### Chat Models
- **GPT-4 Turbo**: `gpt-4-turbo-preview` (128k context)
- **GPT-4**: `gpt-4` (8k context)
- **GPT-4-32k**: `gpt-4-32k` (32k context)
- **GPT-3.5 Turbo**: `gpt-3.5-turbo` (16k context)
- **GPT-3.5 Turbo-16k**: `gpt-3.5-turbo-16k`

### Embedding Models
- **Ada v2**: `text-embedding-ada-002`
- **Small**: `text-embedding-3-small`
- **Large**: `text-embedding-3-large`

## Usage Examples

### Basic Chat

```python
from models.components.cloud_apis.openai import OpenAIAPI

# Initialize
config = {
    "api_key": "your-api-key",
    "default_model": "gpt-3.5-turbo"
}
api = OpenAIAPI(config)

# Simple chat
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain quantum computing"}
]

response = api.chat(messages)
print(response)
```

### Streaming Responses

```python
# Stream for real-time output
for chunk in api.chat(messages, stream=True):
    print(chunk, end="", flush=True)
```

### Advanced Parameters

```python
response = api.chat(
    messages,
    model="gpt-4",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stop=["\n\n", "END"]
)
```

### Function Calling

```python
# Define functions
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

# Chat with function calling
response = api.chat(
    messages,
    tools=tools,
    tool_choice="auto"
)
```

### Embeddings

```python
# Create embeddings
text = "Machine learning is fascinating"
embedding = api.create_embedding(text)

# Batch embeddings
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = api.create_embedding(texts)
```

### Content Moderation

```python
# Check content safety
text = "Some text to check"
moderation = api.moderate_content(text)

if moderation["flagged"]:
    print("Content flagged!")
    print("Categories:", moderation["categories"])
```

### Token Counting

```python
# Count tokens before sending
text = "Your long text here..."
token_count = api.count_tokens(text, model="gpt-4")
print(f"Token count: {token_count}")

# Estimate cost
cost_per_1k_tokens = 0.03  # GPT-4 pricing
estimated_cost = (token_count / 1000) * cost_per_1k_tokens
```

## Best Practices

### 1. API Key Security
```python
# Use environment variables
os.environ["OPENAI_API_KEY"] = "your-key"
config = {"api_key": os.getenv("OPENAI_API_KEY")}
```

### 2. Error Handling
```python
try:
    response = api.chat(messages)
except Exception as e:
    if "rate_limit" in str(e):
        # Handle rate limiting
        time.sleep(60)
    elif "invalid_api_key" in str(e):
        # Handle authentication
        pass
```

### 3. Cost Optimization
- Use `gpt-3.5-turbo` for most tasks
- Set appropriate `max_tokens`
- Use system messages efficiently
- Cache responses when possible

### 4. Context Management
```python
# Trim conversation history
def trim_messages(messages, max_tokens=4000):
    # Keep system message
    trimmed = [messages[0]] if messages[0]["role"] == "system" else []
    
    # Add recent messages up to token limit
    token_count = 0
    for msg in reversed(messages[1:]):
        msg_tokens = api.count_tokens(msg["content"])
        if token_count + msg_tokens > max_tokens:
            break
        trimmed.insert(1, msg)
        token_count += msg_tokens
    
    return trimmed
```

## Rate Limits

| Model | RPM | TPM | Batch Queue Limit |
|-------|-----|-----|-------------------|
| GPT-4 | 500 | 10,000 | 100,000 |
| GPT-4-32k | 500 | 10,000 | 100,000 |
| GPT-3.5 Turbo | 3,500 | 90,000 | 200,000 |

## Pricing (as of 2024)

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-4 | $0.03 | $0.06 |
| GPT-3.5 Turbo | $0.0005 | $0.0015 |
| Embeddings | $0.0001 | N/A |

## Troubleshooting

### Common Issues

1. **Rate Limiting**
   - Implement exponential backoff
   - Use batch processing
   - Upgrade tier if needed

2. **Context Length Errors**
   - Trim messages to fit context
   - Use larger context models
   - Summarize long conversations

3. **API Errors**
   - Check API status: https://status.openai.com
   - Verify API key and organization
   - Check quota and billing

## Integration Tips

- **With RAG**: Use for query understanding and answer generation
- **With Fine-tuning**: Compare fine-tuned vs base model performance
- **With Agents**: Use function calling for tool integration
- **With Evaluation**: Use as a judge for model outputs