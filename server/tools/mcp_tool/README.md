# MCP Tool Integration

This module provides integration with Model Context Protocol (MCP) servers using the official [Python MCP SDK](https://github.com/modelcontextprotocol/python-sdk), allowing LlamaFarm projects to dynamically register and use external tools.

## Overview

The MCP integration consists of three main components:

1. **MCPService** (`services/mcp_service.py`): Manages connections to configured MCP servers using the official Python MCP SDK with `ClientSession`, `stdio_client`, and `sse_client` for robust communication.
2. **MCPToolFactory** (`tools/mcp_tool/tool/mcp_tool_factory.py`): Creates dynamic AtomicAgents tools from MCP server tool schemas.
3. **ProjectChatOrchestratorAgent** integration: Automatically discovers and registers MCP tools when the project is configured with MCP servers.

## Configuration

MCP servers are configured in `llamafarm.yaml` under the `mcp` section:

```yaml
mcp:
  servers:
    - name: my-http-server
      transport: http
      base_url: http://localhost:8080
      headers:
        Authorization: Bearer ${env:MCP_TOKEN}
    
    - name: my-stdio-server
      transport: stdio
      command: python
      args:
        - -m
        - my_mcp_server
      env:
        MCP_ENV: production
```

### Configuration Fields

- `name`: Unique identifier for the MCP server
- `transport`: Either `http` or `stdio`
- For HTTP transport:
  - `base_url`: Base URL of the MCP server
  - `headers`: Optional HTTP headers (supports environment variable interpolation)
- For STDIO transport:
  - `command`: Executable command
  - `args`: Optional command arguments
  - `env`: Optional environment variables

## How It Works

1. When a `ProjectChatOrchestratorAgent` is initialized with an MCP-configured project:
   - The `MCPService` connects to all configured MCP servers
   - The `MCPToolFactory` discovers available tools from each server
   - Dynamic tool wrappers are created for each discovered tool
   - Tools are registered with the agent (if the model supports structured output)

2. During chat sessions:
   - The LLM can discover and call MCP tools like any other tool
   - Tool calls are routed through `MCPService` to the appropriate server
   - Results are returned to the LLM for further processing

## MCP Protocol

This implementation uses the official Python MCP SDK, which handles all protocol details automatically.

### HTTP Transport (SSE)

HTTP MCP servers are accessed via Server-Sent Events (SSE) using the SDK's `sse_client`. The SDK handles:
- Connection management
- Message serialization/deserialization
- Session initialization
- Tool discovery and execution

### STDIO Transport

STDIO MCP servers are accessed via standard input/output using the SDK's `stdio_client` and `StdioServerParameters`. The SDK handles:
- Process spawning and management
- JSON-RPC 2.0 protocol
- Stream management
- Session lifecycle

For details on implementing MCP servers, see the [official MCP documentation](https://modelcontextprotocol.io/).

## Testing

Unit tests for MCP integration are in `server/tests/test_mcp_service.py` and `server/tests/test_mcp_tool_factory.py`.

Run tests with:

```bash
cd server
uv run --group test pytest tests/test_mcp_service.py tests/test_mcp_tool_factory.py -v
```

## Caching

Tool schemas are cached for 5 minutes to reduce unnecessary server calls. Cache is per-server and automatically refreshed when expired.

## Error Handling

- If an MCP server is unavailable, the agent initializes without those tools (logged as warning)
- If a tool call fails, the error is returned to the LLM with details
- Invalid tool schemas are skipped gracefully during discovery

## Example

See the commented example in `server/seeds/project_seed/llamafarm.yaml` for a complete configuration example.

