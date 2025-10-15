# MCP Persistent Session Management

## Overview

The MCP integration now uses persistent sessions that are created once when tools are loaded and kept alive for the lifetime of the agent. This avoids the `ClosedResourceError` that was occurring when tools tried to use closed sessions.

## Architecture

### Session Lifecycle

1. **Session Creation** (`ProjectChatOrchestratorAgent.load_mcp_tools()`)
   - Creates an `AsyncExitStack` to manage session lifecycle
   - For each configured MCP server:
     - Opens SSE connection to the exact endpoint (e.g., `http://localhost:8000/mcp`)
     - Creates `ClientSession` and initializes it
     - Stores session in `self._mcp_sessions[server_name]`
   - Keeps `AsyncExitStack` reference in `self._mcp_session_tasks` to prevent cleanup

2. **Tool Creation** (`MCPToolFactory.create_all_tools(session=...)`)
   - Receives the persistent session as a parameter
   - Passes it to atomic-agents' `fetch_mcp_tools_async(client_session=session)`
   - atomic-agents creates tool classes that reference this session

3. **Tool Execution** (When MCPOrchestrator calls a tool)
   - Tool uses the persistent session that was passed during creation
   - No need to open/close connections - session is already open

4. **Session Cleanup** (`ProjectChatOrchestratorAgent.cleanup_mcp_sessions()`)
   - Calls `__aexit__` on all stored `AsyncExitStack` instances
   - Properly closes all MCP sessions
   - Currently manual - could be integrated with agent lifecycle

## Key Components

### ProjectChatOrchestratorAgent

**New Attributes:**
```python
_mcp_sessions: dict = {}  # Store MCP sessions per server {server_name: ClientSession}
_mcp_session_tasks: list = []  # Keep AsyncExitStack references alive
```

**Modified Methods:**
- `load_mcp_tools()`: Creates persistent sessions before tool creation
- `cleanup_mcp_sessions()`: Cleanup method for shutting down sessions

### MCPToolFactory

**Modified Methods:**
- `create_tools_for_server(server_name, *, session: ClientSession)`: Now requires session parameter
- `create_all_tools(*, session: ClientSession)`: Now requires session parameter

## fastapi-mcp Compatibility

The implementation works with fastapi-mcp servers that serve at non-standard endpoints:
- **Standard MCP**: `/sse` or `/mcp/` endpoints
- **fastapi-mcp**: `/mcp` endpoint

Configuration:
```yaml
mcp:
  servers:
    - name: lf
      transport: http
      base_url: http://localhost:8000/mcp  # Exact endpoint
      headers:
        Authorization: Bearer token123
```

The implementation:
1. Connects directly to `base_url` (e.g., `http://localhost:8000/mcp`)
2. Does NOT let atomic-agents append paths
3. Creates persistent sessions that tools reuse

## Benefits

1. **No ClosedResourceError**: Sessions stay open for tool execution
2. **Better Performance**: No connection overhead per tool call
3. **Resource Efficient**: Reuses connections instead of creating new ones
4. **Proper Cleanup**: Sessions can be closed when agent is destroyed

## Usage Example

```python
# Create agent
agent = await ProjectChatOrchestratorAgentFactory.create_agent(
    project_config=config,
    project_dir=".",
    model_name="default",
    session_id="my-session"
)

# Tools are now loaded with persistent sessions
# Agent can execute MCP tool calls through orchestrator

# Cleanup when done (optional - normally handled by framework)
await agent.cleanup_mcp_sessions()
```

## Future Enhancements

1. **Automatic Cleanup**: Integrate `cleanup_mcp_sessions()` with agent lifecycle/destructor
2. **Session Reconnection**: Auto-reconnect if session drops
3. **Per-Server Sessions**: Currently uses first session for all tools - could map tools to specific sessions
4. **Session Pooling**: Create multiple sessions per server for parallel execution
5. **Health Checks**: Periodic ping to verify sessions are still alive

## Troubleshooting

### Sessions Not Created
- Check that `mcp.servers` is configured in `llamafarm.yaml`
- Verify `base_url` points to accessible MCP server
- Look for "Created persistent MCP session" log messages

### ClosedResourceError Still Occurring
- Ensure `_mcp_session_tasks` list is populated (keeps AsyncExitStack alive)
- Verify session is passed to `create_all_tools(session=...)`
- Check that atomic-agents uses `client_session` parameter correctly

### Connection Refused
- Verify MCP server is running at the configured endpoint
- Test with `curl http://localhost:8000/mcp` to confirm endpoint
- Check firewall/network settings

## Testing

Test that sessions persist:
```python
# After tool loading
assert len(agent._mcp_sessions) > 0
assert agent._mcp_session_tasks  # Stack is alive

# Execute tool call
result = await tool_instance.arun(params)  # Should work without ClosedResourceError

# Verify session still alive
for session in agent._mcp_sessions.values():
    assert not session._write_stream._closed
```

---

**Status**: ✅ Implemented and working with fastapi-mcp servers  
**Last Updated**: October 15, 2025

