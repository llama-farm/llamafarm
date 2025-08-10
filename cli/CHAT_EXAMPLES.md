# LlamaFarm CLI Chat Examples

## Starting a Chat Session

The `lf projects chat` command provides an interactive chat interface that connects to your LlamaFarm server.

### Basic Usage

```bash
# Start a chat session using defaults from llamafarm.yaml
lf projects chat

# If your llamafarm.yaml contains:
# name: my-org/my-project
# This will automatically use:
# - Server: http://localhost:8000 (default)
# - Namespace: my-org (from config)
# - Project: my-project (from config)
```

### Override Defaults

```bash
# Override server URL while keeping config project settings
lf projects chat --server-url http://prod-server:8000

# Override specific project settings
lf projects chat --namespace my-org --project my-project

# Use a different config file
lf projects chat --config /path/to/other-config.yaml

# Full override with custom parameters
lf projects chat \
  --server-url http://localhost:8000 \
  --namespace my-org \
  --project my-project \
  --temperature 0.8 \
  --max-tokens 2000 \
  --session-id existing-session-123
```

## Chat Interface Commands

Once in the chat session, you can use these commands:

- `help` - Show available commands and current settings
- `clear` - Clear conversation history and start a new session
- `exit` or `quit` - Exit the chat session

## API Endpoint

The chat command sends requests to:
```
POST /v1/projects/{namespace}/{project_id}/chat/completions
```

### Request Format

```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### Response Format

```json
{
  "id": "chat-12345",
  "object": "chat.completion",
  "created": 1640995200,
  "model": "llama3.1:8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Features

### ✅ Implemented
- Interactive terminal chat interface
- OpenAI-compatible chat completions API format
- Conversation history maintained locally
- Session management with X-Session-ID header
- Configurable temperature and max tokens
- Error handling for network issues
- Help system and special commands
- Clean exit functionality

### 🔄 Session Management
- Automatic session ID generation if not provided
- Session continuity across multiple messages
- Clear command to reset conversation history

### ⚙️ Configuration
- Server URL (default: http://localhost:8000)
- Project namespace and ID (from llamafarm.yaml name field or flags)
- Temperature control (default: 0.7)
- Max tokens limit (default: 1000)
- Optional session ID for continuing conversations
- Config file path (default: llamafarm.yaml in current directory)

## Example Session

```
$ lf projects chat

🌾 Starting LlamaFarm chat session...
📡 Server: http://localhost:8000
📁 Project: my-namespace/my-project

Type 'exit' or 'quit' to end the session, 'clear' to start a new session.
Type your message and press Enter to send.

You: What can you help me with?
Assistant: I'm an AI assistant for the my-namespace/my-project. I can help you with questions about your documents, provide summaries, and assist with various tasks based on the knowledge in your project's dataset.

You: help
Available commands:
  help  - Show this help message
  clear - Clear conversation history and start a new session
  exit  - Exit the chat session
  quit  - Exit the chat session

Chat parameters:
  Temperature: 0.7
  Max tokens: 1000
  Server: http://localhost:8000
  Project: my-namespace/my-project

You: exit
👋 Goodbye!
```