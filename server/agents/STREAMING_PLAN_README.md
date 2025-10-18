# Streaming Tool Calling - Implementation Plan

## Overview

This plan implements streaming responses with MCP tool calling support. The design is clean, provider-agnostic, and maintains proper abstraction layers.

## Key Principles

1. **No backwards compatibility** - Clean, optimal design for a new project
2. **Consistent interface** - All clients return identical response format
3. **Proper abstraction** - ChatOrchestrator never touches clients directly  
4. **Provider-agnostic** - LFAgent doesn't know how tools are injected

## Architecture

```
ChatOrchestratorAgent
  - Manages MCP tool execution loop
  - Converts MCP tools → ToolDefinition
  - Executes tools when requested
        ↓ calls
LFAgent (base class)
  - stream_chat_with_tools(tools, user_input)
  - Prepares messages, delegates to client
  - Returns consistent StreamEvent format
        ↓ delegates to
LFAgentClient (abstract)
  - Each impl handles tool injection differently
  - Returns consistent StreamEvent format
    ├─ LFAgentClientOpenAI (native function calling)
    └─ LFAgentClientOllama (JSON prompting)
```

## Core Design

### Unified Response Format

All clients return `StreamEvent` objects:
```python
@dataclass
class StreamEvent:
    type: Literal["content", "tool_call"]
    content: str | None = None
    tool_call: ToolCallRequest | None = None
```

### Client Implementations Hide Differences

**OpenAI Client**:
- Uses OpenAI `tools` parameter (native function calling)
- Parses `tool_calls` from stream
- Returns `StreamEvent` objects

**Ollama Client**:
- Injects tools into system prompt
- Detects JSON responses
- Returns `StreamEvent` objects (same format!)

### Orchestrator is Provider-Agnostic

```python
async for event in self.stream_chat_with_tools(user_input=input, tools=tools):
    if event.is_content():
        yield event.content  # Stream to user
    elif event.is_tool_call():
        result = await self._execute_mcp_tool(...)  # Execute tool
        # Continue loop for final answer
```

The orchestrator doesn't know or care how the client implements tool calling.

## Implementation

See **`STREAMING_ARCHITECTURE_V2.md`** for:
- Complete code examples
- All type definitions
- Full client implementations
- Orchestrator updates
- Step-by-step implementation plan

## Files to Modify

### Create New
1. `agents/llamagent/types.py` - `ToolDefinition`, `StreamEvent`, `ToolCallRequest`

### Modify
2. `agents/llamagent/clients/client.py` - Add `stream_chat_with_tools()` abstract method
3. `agents/llamagent/clients/openai.py` - Implement native function calling
4. `agents/llamagent/clients/ollama.py` - Implement JSON-based tool calling
5. `agents/llamagent/agent.py` - Add `stream_chat_with_tools()` method
6. `agents/chat_orchestrator.py` - Use LFAgent interface, never touch client

### Delete
7. `context_providers/mcp_tools_context_provider.py` - No longer needed

## Timeline

- **Phase 1**: Types & Base (0.5 day)
- **Phase 2**: OpenAI Client (1 day)
- **Phase 3**: Ollama Client (1 day)  
- **Phase 4**: Orchestrator (1 day)
- **Phase 5**: Testing (0.5 day)

**Total: 4 days**

## Benefits

✅ Clean separation of concerns  
✅ Easy to test (mock `StreamEvent` responses)  
✅ Easy to extend (new provider = implement interface)  
✅ Optimal UX (native function calling where available)  
✅ Universal compatibility (JSON fallback)  
✅ Provider-agnostic orchestrator  

## Next Steps

1. Review `STREAMING_ARCHITECTURE_V2.md` for complete implementation details
2. Start with Phase 1: Create types and update base interfaces
3. Implement client-specific logic (Phases 2-3)
4. Update orchestrator to use clean abstraction (Phase 4)
5. Test and validate (Phase 5)

