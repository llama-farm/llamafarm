---
title: MCP (Model Context Protocol)
sidebar_position: 6
---

# MCP (Model Context Protocol)

MCP is a standardized protocol that gives AI models access to external tools, APIs, and data sources. LlamaFarm supports MCP both as a **client** (connecting to external MCP servers) and as a **server** (exposing its own API as MCP tools).

## Why MCP?

Instead of limiting your AI to text generation, MCP lets you connect models to:

- **File systems** - Read and analyze local files
- **Databases** - Query and modify data
- **APIs** - Interact with external services (weather, CRM, calendars)
- **Custom tools** - Expose your own business logic

## Quick Start

### 1. Add an MCP Server to Your Project

In your `llamafarm.yaml`:

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args:
        - '-y'
        - '@modelcontextprotocol/server-filesystem'
        - '/path/to/allowed/directory'
```

### 2. Assign It to a Model

```yaml
runtime:
  models:
    - name: assistant
      provider: ollama
      model: llama3.1:8b
      mcp_servers:
        - filesystem
```

### 3. Chat with Tool Access

```bash
lf chat "What files are in my documents folder?"
```

The model can now use the filesystem tools to answer your question.

---

## Transport Types

LlamaFarm supports three transport mechanisms for connecting to MCP servers:

### STDIO (Local Process)

Spawns the MCP server as a local subprocess. Best for local tools.

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args:
        - '-y'
        - '@modelcontextprotocol/server-filesystem'
        - '/Users/myuser/documents'
      env:
        NODE_ENV: production
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | string | Yes | Command to launch the server |
| `args` | array | No | Arguments to pass to the command |
| `env` | object | No | Environment variables for the process |

**Common STDIO Servers:**

```yaml
# Official Anthropic filesystem server
- name: filesystem
  transport: stdio
  command: npx
  args: ['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir']

# SQLite database server
- name: sqlite
  transport: stdio
  command: npx
  args: ['-y', '@modelcontextprotocol/server-sqlite', 'path/to/database.db']

# Custom Python MCP server
- name: my-python-tool
  transport: stdio
  command: python
  args: ['-m', 'my_mcp_server']
```

### HTTP (Remote Server)

Connects to a remote MCP server via HTTP streaming. Best for cloud services and shared tools.

```yaml
mcp:
  servers:
    - name: remote-api
      transport: http
      base_url: https://api.example.com/mcp
      headers:
        Authorization: Bearer ${env:API_TOKEN}
        X-Custom-Header: my-value
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_url` | string | Yes | Base URL of the MCP server |
| `headers` | object | No | HTTP headers (supports `${env:VAR}` substitution) |

**Use Cases:**
- FastAPI-MCP servers
- Cloud-hosted tool services
- Shared team tools

### SSE (Server-Sent Events)

Connects via Server-Sent Events for real-time streaming.

```yaml
mcp:
  servers:
    - name: streaming-server
      transport: sse
      base_url: https://api.example.com/mcp/sse
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `base_url` | string | Yes | Base URL of the SSE endpoint |

---

## Per-Model Tool Access

Control which models can access which MCP servers. This is essential for security and capability management.

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/data']

    - name: database
      transport: http
      base_url: http://localhost:8080/mcp

runtime:
  models:
    # Research assistant - read-only file access
    - name: research-assistant
      provider: openai
      model: gpt-4
      mcp_servers:
        - filesystem

    # Data analyst - database access only
    - name: data-analyst
      provider: openai
      model: gpt-4
      mcp_servers:
        - database

    # General chat - no tool access (safer)
    - name: general-chat
      provider: ollama
      model: llama3.1:8b
      mcp_servers: []  # Empty list = no MCP
```

### Access Control Rules

| Configuration | Behavior |
|---------------|----------|
| `mcp_servers: [server1, server2]` | Model can only use listed servers |
| `mcp_servers: []` | Model has no MCP access |
| `mcp_servers` omitted | Model can use all configured servers |

---

## LlamaFarm as an MCP Server

LlamaFarm itself exposes its API as MCP tools, allowing external clients (like Claude Desktop or Cursor) to control it.

### Connecting from Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "llamafarm": {
      "transport": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Or use the HTTP transport if your MCP client supports it:

```json
{
  "mcpServers": {
    "llamafarm": {
      "transport": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Available Tools (Exposed by LlamaFarm)

When you connect to LlamaFarm as an MCP server, you get access to:

| Tool | Description |
|------|-------------|
| List projects | Get all projects in a namespace |
| Create project | Create a new LlamaFarm project |
| Get project | Get project details and configuration |
| Update project | Modify project settings |
| Delete project | Remove a project |
| Chat completions | Send messages to AI models |
| RAG query | Query documents in vector databases |
| List models | Get available AI models |

### Self-Reference Configuration

You can configure your project to use its own LlamaFarm server as an MCP source:

```yaml
mcp:
  servers:
    - name: llamafarm-server
      transport: http
      base_url: http://localhost:8000/mcp

runtime:
  models:
    - name: default
      provider: ollama
      model: llama3.1:8b
      mcp_servers: [llamafarm-server]
```

This allows models to interact with other LlamaFarm features programmatically.

---

## Tool Call Strategies

LlamaFarm supports two strategies for tool calling:

### Native API (Recommended)

Uses the model provider's native tool calling capabilities.

```yaml
runtime:
  models:
    - name: assistant
      provider: openai
      model: gpt-4
      tool_call_strategy: native_api
```

**Supported providers:** OpenAI, Anthropic, most modern LLM APIs

### Prompt-Based

Injects tool definitions into the system prompt for models that don't support native tool calling.

```yaml
runtime:
  models:
    - name: assistant
      provider: ollama
      model: llama2
      tool_call_strategy: prompt_based
```

**Use when:** Using older models or providers without native tool support

---

## Complete Configuration Example

Here's a full `llamafarm.yaml` with MCP configured:

```yaml
version: v1
name: my-project
namespace: default

mcp:
  servers:
    # Local filesystem access
    - name: filesystem
      transport: stdio
      command: npx
      args:
        - '-y'
        - '@modelcontextprotocol/server-filesystem'
        - '/Users/myuser/projects'

    # SQLite database access
    - name: database
      transport: stdio
      command: npx
      args:
        - '-y'
        - '@modelcontextprotocol/server-sqlite'
        - './data/app.db'

    # Remote API with authentication
    - name: company-api
      transport: http
      base_url: https://api.mycompany.com/mcp
      headers:
        Authorization: Bearer ${env:COMPANY_API_KEY}

    # LlamaFarm's own API
    - name: llamafarm
      transport: http
      base_url: http://localhost:8000/mcp

runtime:
  default_model: assistant
  models:
    # Full-featured assistant with all tools
    - name: assistant
      provider: openai
      model: gpt-4
      tool_call_strategy: native_api
      mcp_servers:
        - filesystem
        - database
        - company-api

    # Restricted model for public use
    - name: public-chat
      provider: ollama
      model: llama3.1:8b
      mcp_servers: []  # No tool access

    # Internal tool for automation
    - name: automation
      provider: openai
      model: gpt-4
      mcp_servers:
        - llamafarm
        - database

prompts:
  - name: default
    messages:
      - role: system
        content: |
          You are a helpful assistant with access to various tools.
          Use the available tools to help answer questions and complete tasks.
```

---

## Environment Variable Substitution

Use `${env:VARIABLE_NAME}` to inject environment variables into your configuration:

```yaml
mcp:
  servers:
    - name: secure-api
      transport: http
      base_url: ${env:API_BASE_URL}
      headers:
        Authorization: Bearer ${env:API_TOKEN}
        X-API-Key: ${env:API_KEY}
```

This keeps secrets out of your configuration files.

---

## Session Management

LlamaFarm maintains persistent MCP sessions for better performance:

- **Connection pooling** - Sessions are reused across requests
- **5-minute cache** - Tool lists are cached to reduce overhead
- **Graceful shutdown** - Sessions are properly closed when the server stops
- **1-hour timeout** - Long-running sessions for persistent connections

---

## Debugging MCP

### Check Server Configuration

```bash
lf config show | grep -A 20 "mcp:"
```

### Test MCP Connection

The server logs will show MCP initialization:

```
MCPService initialized [services.mcp_service] server_count=2
Created all MCP tools [tools.mcp_tool] servers=['filesystem', 'database'] total_tools=5
MCP tools loaded [agents.chat_orchestrator] tool_count=5 tool_names=['read_file', 'write_file', ...]
```

### Common Issues

| Issue | Solution |
|-------|----------|
| "Server not found" | Check server name matches in `mcp_servers` list |
| "Connection refused" | Ensure HTTP server is running at `base_url` |
| "Command not found" | Install required packages (e.g., `npm install -g @modelcontextprotocol/server-filesystem`) |
| "Permission denied" | Check file/directory permissions for STDIO servers |

---

## Real-World Use Cases

### Code Analysis Assistant

```yaml
mcp:
  servers:
    - name: codebase
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', './src']

runtime:
  models:
    - name: code-reviewer
      provider: openai
      model: gpt-4
      mcp_servers: [codebase]
      prompts: [code-review]

prompts:
  - name: code-review
    messages:
      - role: system
        content: |
          You are a senior software engineer reviewing code.
          Use the filesystem tools to read and analyze source files.
          Provide constructive feedback on code quality, patterns, and potential issues.
```

### Database Query Assistant

```yaml
mcp:
  servers:
    - name: analytics-db
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-sqlite', './analytics.db']

runtime:
  models:
    - name: data-analyst
      provider: openai
      model: gpt-4
      mcp_servers: [analytics-db]
```

### Multi-Tool Automation

```yaml
mcp:
  servers:
    - name: filesystem
      transport: stdio
      command: npx
      args: ['-y', '@modelcontextprotocol/server-filesystem', '/data']

    - name: github
      transport: http
      base_url: https://github-mcp.example.com/mcp
      headers:
        Authorization: Bearer ${env:GITHUB_TOKEN}

    - name: slack
      transport: http
      base_url: https://slack-mcp.example.com/mcp
      headers:
        Authorization: Bearer ${env:SLACK_TOKEN}

runtime:
  models:
    - name: devops-bot
      provider: openai
      model: gpt-4
      mcp_servers: [filesystem, github, slack]
```

---

## Building Your Own MCP Server

You can create custom MCP servers to expose your own tools. See the [MCP documentation](https://modelcontextprotocol.io/) for the full specification.

### Example: Python MCP Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("my-tools")

@app.tool()
async def my_custom_tool(param1: str, param2: int) -> str:
    """Description of what this tool does."""
    # Your implementation
    return f"Result: {param1}, {param2}"

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Then configure it in LlamaFarm:

```yaml
mcp:
  servers:
    - name: my-tools
      transport: stdio
      command: python
      args: ['-m', 'my_mcp_server']
```

---

## Next Steps

- [Configuration Guide](../configuration/index.md) - Full configuration reference
- [Models & Runtime](../models/index.md) - Configure AI models
- [RAG Guide](../rag/index.md) - Set up document retrieval
- [MCP Specification](https://modelcontextprotocol.io/) - Official MCP documentation
