# Streaming Tool Calling Architecture - Clean Design

## Design Principles

1. **No backwards compatibility** - Clean break, optimal design
2. **Consistent interface** - All clients return the same response format
3. **Proper abstraction** - ChatOrchestrator never touches clients directly
4. **Provider-agnostic** - LFAgent doesn't know how tools are injected

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│ ChatOrchestratorAgent                           │
│ - Manages MCP tool execution loop               │
│ - Converts MCP tools → Tool definitions         │
│ - Executes tools when requested                 │
│ - Manages conversation history                  │
└─────────────────────────────────────────────────┘
                    ↓ calls
┌─────────────────────────────────────────────────┐
│ LFAgent (base class)                            │
│ - stream_chat_with_tools(tools, user_input)     │
│ - Prepares messages (system prompt + history)   │
│ - Delegates to client                           │
│ - Returns consistent StreamEvent format         │
└─────────────────────────────────────────────────┘
                    ↓ delegates to
┌─────────────────────────────────────────────────┐
│ LFAgentClient (abstract)                        │
│ - stream_chat_with_tools(messages, tools)       │
│ - Each impl handles tool injection differently  │
│ - Returns consistent StreamEvent format         │
│   ├─ LFAgentClientOpenAI (native function call) │
│   └─ LFAgentClientOllama (JSON prompting)       │
└─────────────────────────────────────────────────┘
```

---

## Core Types

### Tool Definition

```python
# agents/llamagent/types.py

from dataclasses import dataclass
from typing import Any

@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition"""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for parameters
    
    @classmethod
    def from_mcp_tool(cls, tool_class: type) -> "ToolDefinition":
        """Convert MCP tool class to ToolDefinition"""
        tool_name = getattr(tool_class, "mcp_tool_name", tool_class.__name__)
        tool_description = tool_class.__doc__ or "No description"
        
        # Get input schema from tool
        input_schema_class = getattr(tool_class, "input_schema", None)
        if input_schema_class:
            schema = input_schema_class.model_json_schema()
            # Remove tool_name discriminator field
            props = {k: v for k, v in schema.get("properties", {}).items() 
                     if k != "tool_name"}
            required = [r for r in schema.get("required", []) if r != "tool_name"]
            parameters = {
                "type": "object",
                "properties": props,
                "required": required
            }
        else:
            parameters = {"type": "object", "properties": {}}
        
        return cls(
            name=tool_name,
            description=tool_description,
            parameters=parameters
        )
```

### Stream Events

```python
from dataclasses import dataclass
from typing import Any, Literal

@dataclass
class ToolCallRequest:
    """A tool call requested by the LLM"""
    id: str  # Unique ID for this tool call (for tracking)
    name: str
    arguments: dict[str, Any]

@dataclass
class StreamEvent:
    """Event from streaming chat"""
    type: Literal["content", "tool_call"]
    
    # For type="content"
    content: str | None = None
    
    # For type="tool_call"
    tool_call: ToolCallRequest | None = None
    
    def is_content(self) -> bool:
        return self.type == "content"
    
    def is_tool_call(self) -> bool:
        return self.type == "tool_call"
```

---

## LFAgentClient Interface

### Abstract Base Class

```python
# agents/llamagent/clients/client.py

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from agents.llamagent.history import LFAgentChatMessage
from agents.llamagent.types import StreamEvent, ToolDefinition

class LFAgentClient(ABC):
    """Abstract base class for LLM clients.
    
    Each implementation handles tool calling in their own way:
    - OpenAI: Uses native `tools` parameter
    - Ollama: Injects tools into system prompt, detects JSON
    - Others: Whatever works for that provider
    
    All implementations must return the same StreamEvent format.
    """
    
    def __init__(self, *, model_config: Model):
        self._model_name = model_config.name
        self._model_config = model_config
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @staticmethod
    @abstractmethod
    def prompt_to_message(prompt: Prompt) -> LFAgentChatMessage:
        """Convert a llamafarm Prompt to a LFAgentChatMessage."""
        pass
    
    @abstractmethod
    async def stream_chat_with_tools(
        self,
        *,
        messages: list[LFAgentChatMessage],
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with tool calling support.
        
        The implementation is responsible for:
        1. Injecting tools (via API param, system prompt, etc.)
        2. Detecting tool call requests in response
        3. Yielding StreamEvent objects in consistent format
        
        Args:
            messages: Conversation history
            tools: Available tools (provider-agnostic format)
        
        Yields:
            StreamEvent: Either content chunks or tool call requests
        """
        pass
```

---

## Client Implementations

### OpenAI Client (Native Function Calling)

```python
# agents/llamagent/clients/openai.py

from collections.abc import AsyncGenerator
import json

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agents.llamagent.history import LFAgentChatMessage
from agents.llamagent.types import StreamEvent, ToolCallRequest, ToolDefinition
from core.logging import FastAPIStructLogger

from .client import LFAgentClient

logger = FastAPIStructLogger(__name__)


class LFAgentClientOpenAI(LFAgentClient):
    """OpenAI client using native function calling.
    
    This client:
    1. Passes tools via the `tools` API parameter
    2. Detects tool calls from native `tool_calls` in response
    3. Streams both content and tool calls as StreamEvents
    """
    
    async def stream_chat_with_tools(
        self,
        *,
        messages: list[LFAgentChatMessage],
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with native OpenAI function calling."""
        
        client = AsyncOpenAI(
            api_key=self._model_config.api_key or "",
            base_url=self._model_config.base_url or "",
        )
        
        # Convert tools to OpenAI format
        openai_tools = [self._tool_to_openai_format(t) for t in tools] if tools else None
        
        # Create streaming request
        response_stream = await client.chat.completions.create(
            model=self._model_config.model,
            messages=[self._message_to_openai_message(m) for m in messages],
            tools=openai_tools,
            tool_choice="auto" if openai_tools else None,
            stream=True,
            **(self._model_config.model_api_parameters or {}),
        )
        
        # Track partial tool calls as they stream in
        current_tool_calls: dict[int, dict] = {}
        
        async for chunk in response_stream:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            # Yield content chunks
            if delta.content:
                yield StreamEvent(type="content", content=delta.content)
            
            # Handle tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    
                    # Initialize tool call if new
                    if idx not in current_tool_calls:
                        current_tool_calls[idx] = {
                            "id": tc_delta.id or f"call_{idx}",
                            "name": "",
                            "arguments": ""
                        }
                    
                    # Accumulate name
                    if tc_delta.function and tc_delta.function.name:
                        current_tool_calls[idx]["name"] = tc_delta.function.name
                    
                    # Accumulate arguments
                    if tc_delta.function and tc_delta.function.arguments:
                        current_tool_calls[idx]["arguments"] += tc_delta.function.arguments
            
            # When stream finishes with tool calls, yield them
            if choice.finish_reason == "tool_calls":
                for tc_data in current_tool_calls.values():
                    try:
                        args = json.loads(tc_data["arguments"])
                        yield StreamEvent(
                            type="tool_call",
                            tool_call=ToolCallRequest(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=args
                            )
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            "Failed to parse tool call arguments",
                            arguments=tc_data["arguments"],
                            error=str(e)
                        )
    
    @staticmethod
    def prompt_to_message(prompt: Prompt) -> LFAgentChatMessage:
        return LFAgentChatMessage(role="system", content=prompt.content)
    
    def _tool_to_openai_format(self, tool: ToolDefinition) -> dict:
        """Convert ToolDefinition to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
    
    def _message_to_openai_message(
        self, message: LFAgentChatMessage
    ) -> ChatCompletionMessageParam:
        """Convert LFAgentChatMessage to OpenAI format."""
        match message.role:
            case "system":
                return {"role": "system", "content": message.content}
            case "user":
                return {"role": "user", "content": message.content}
            case "assistant":
                return {"role": "assistant", "content": message.content}
            case "tool":
                # For tool results, format as user message with result
                return {"role": "user", "content": message.content}
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
```

### Ollama Client (JSON-Based)

```python
# agents/llamagent/clients/ollama.py

from collections.abc import AsyncGenerator
import json

from ollama import AsyncClient, Message

from agents.llamagent.history import LFAgentChatMessage
from agents.llamagent.types import StreamEvent, ToolCallRequest, ToolDefinition
from core.logging import FastAPIStructLogger

from .client import LFAgentClient

logger = FastAPIStructLogger(__name__)


class LFAgentClientOllama(LFAgentClient):
    """Ollama client using JSON-based tool calling.
    
    This client:
    1. Injects tools into the system prompt as instructions + schemas
    2. Detects JSON responses that match tool call format
    3. Yields StreamEvents in the same format as OpenAI client
    """
    
    async def stream_chat_with_tools(
        self,
        *,
        messages: list[LFAgentChatMessage],
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with JSON-based tool calling."""
        
        # Inject tools into system message
        if tools:
            tool_instruction = self._create_tool_instruction(tools)
            
            # Prepend tool instruction to first system message or create new one
            modified_messages = []
            system_injected = False
            for msg in messages:
                if msg.role == "system" and not system_injected:
                    # Prepend to existing system message
                    modified_msg = LFAgentChatMessage(
                        role="system",
                        content=f"{tool_instruction}\n\n{msg.content}"
                    )
                    modified_messages.append(modified_msg)
                    system_injected = True
                else:
                    modified_messages.append(msg)
            
            # If no system message, create one
            if not system_injected:
                modified_messages.insert(
                    0,
                    LFAgentChatMessage(role="system", content=tool_instruction)
                )
            
            messages = modified_messages
        
        # Stream response
        client = AsyncClient(
            host=(
                self._model_config.base_url.rstrip("/v1")
                if self._model_config.base_url
                else ""
            ),
        )
        
        response_stream = await client.chat(
            model=self._model_config.model,
            messages=[self._message_to_ollama_message(m) for m in messages],
            stream=True,
            **(self._model_config.model_api_parameters or {}),
        )
        
        # Buffer for detecting JSON tool calls
        buffer = ""
        looks_like_json = False
        
        async for chunk in response_stream:
            content = chunk.message.content
            if not content:
                continue
            
            buffer += content
            
            # Detect if this looks like a JSON tool call
            stripped = buffer.strip()
            if not looks_like_json and stripped.startswith("{"):
                looks_like_json = True
            
            # If not JSON, stream content normally
            if not looks_like_json:
                yield StreamEvent(type="content", content=content)
        
        # After stream ends, check if buffer contains tool call
        if looks_like_json:
            try:
                data = json.loads(buffer.strip())
                
                # Check if it matches tool call format
                if "tool_name" in data and "tool_parameters" in data:
                    yield StreamEvent(
                        type="tool_call",
                        tool_call=ToolCallRequest(
                            id=f"call_{data['tool_name']}",
                            name=data["tool_name"],
                            arguments=data.get("tool_parameters", {})
                        )
                    )
                else:
                    # JSON but not a tool call, treat as content
                    yield StreamEvent(type="content", content=buffer)
            except json.JSONDecodeError:
                # Not valid JSON, treat as content
                yield StreamEvent(type="content", content=buffer)
    
    @staticmethod
    def prompt_to_message(prompt: Prompt) -> LFAgentChatMessage:
        return LFAgentChatMessage(role="system", content=prompt.content)
    
    def _create_tool_instruction(self, tools: list[ToolDefinition]) -> str:
        """Create system prompt instructions for JSON-based tool calling."""
        tool_schemas = []
        for tool in tools:
            tool_schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        
        instruction = f"""You have access to the following tools. When you want to use a tool, respond with ONLY a JSON object in this exact format:

{{
  "tool_name": "<tool_name>",
  "tool_parameters": {{<parameters>}}
}}

Do not include any other text or explanation when calling a tool.

Available tools:
{json.dumps(tool_schemas, indent=2)}
"""
        return instruction
    
    def _message_to_ollama_message(self, message: LFAgentChatMessage) -> Message:
        """Convert LFAgentChatMessage to Ollama Message format."""
        match message.role:
            case "system":
                return Message(role="system", content=message.content)
            case "user":
                return Message(role="user", content=message.content)
            case "assistant":
                return Message(role="assistant", content=message.content)
            case "tool":
                # Tool results as user messages
                return Message(role="user", content=message.content)
            case _:
                raise ValueError(f"Unknown message role: {message.role}")
```

---

## LFAgent Updates

### Add Tool Calling Method

```python
# agents/llamagent/agent.py

from collections.abc import AsyncGenerator

from agents.llamagent.clients.client import LFAgentClient
from agents.llamagent.history import LFAgentChatMessage, LFAgentHistory
from agents.llamagent.system_prompt_generator import LFAgentSystemPromptGenerator
from agents.llamagent.types import StreamEvent, ToolDefinition

class LFAgent:
    """Base agent class that provides provider-agnostic interface."""
    
    history: LFAgentHistory
    _system_prompt_generator: LFAgentSystemPromptGenerator
    _client: LFAgentClient
    
    def __init__(self, config: LFAgentConfig):
        self.history = config.history
        self._system_prompt_generator = config.system_prompt_generator
        self._client = config.client
    
    async def stream_chat_with_tools(
        self,
        *,
        user_input: LFAgentChatMessage | None = None,
        tools: list[ToolDefinition],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream chat with tool calling support.
        
        This method is provider-agnostic. The client handles:
        - How tools are injected (API param vs system prompt)
        - How tool calls are detected (native vs JSON)
        
        Args:
            user_input: Optional user message to add to history
            tools: Available tools for the LLM to use
        
        Yields:
            StreamEvent: Content chunks or tool call requests
        """
        if user_input:
            self.history.add_message(user_input)
        
        # Prepare messages (system prompt + history)
        messages = self._prepare_messages()
        
        # Delegate to client - it handles provider-specific logic
        async for event in self._client.stream_chat_with_tools(
            messages=messages,
            tools=tools
        ):
            yield event
    
    def _prepare_messages(self) -> list[LFAgentChatMessage]:
        """Prepare messages for the LLM (system prompt + history)."""
        messages: list[LFAgentChatMessage] = []
        
        # Add system prompt
        system_prompt = self._system_prompt_generator.generate_prompt()
        if system_prompt:
            messages.append(LFAgentChatMessage(role="system", content=system_prompt))
        
        # Add conversation history
        for message in self.history.get_history():
            messages.append(
                LFAgentChatMessage(role=message.role, content=message.content)
            )
        
        return messages
    
    # ... existing methods ...
```

---

## ChatOrchestratorAgent Implementation

```python
# agents/chat_orchestrator.py

from collections.abc import AsyncGenerator
import uuid

from agents.llamagent.agent import LFAgent
from agents.llamagent.history import LFAgentChatMessage
from agents.llamagent.types import ToolDefinition
from core.logging import FastAPIStructLogger
from tools.mcp_tool.tool.mcp_tool_factory import BaseTool

logger = FastAPIStructLogger(__name__)


class ChatOrchestratorAgent(LFAgent):
    """Orchestrator for MCP tool calling.
    
    This class:
    1. Converts MCP tools to ToolDefinition format
    2. Calls LFAgent.stream_chat_with_tools()
    3. Executes tools when LLM requests them
    4. Manages the conversation loop
    """
    
    _mcp_tools: list[type[BaseTool]] = []
    _max_iterations: int = 10
    
    async def run_async_stream(
        self, user_input: LFAgentChatMessage | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat with MCP tool execution."""
        
        if not self._mcp_enabled or not self._mcp_tools:
            # No MCP tools, use standard streaming
            async for chunk in super().run_async_stream(user_input):
                yield chunk
            return
        
        # Convert MCP tools to ToolDefinition format
        tools = [ToolDefinition.from_mcp_tool(t) for t in self._mcp_tools]
        
        iteration = 0
        current_input = user_input
        
        while iteration < self._max_iterations:
            iteration += 1
            
            logger.info("Starting tool calling iteration", iteration=iteration)
            
            # Stream chat with tools
            tool_call_made = False
            
            async for event in self.stream_chat_with_tools(
                user_input=current_input,
                tools=tools
            ):
                if event.is_content():
                    # Stream content to user
                    yield event.content
                
                elif event.is_tool_call():
                    # Execute the tool
                    tool_call_made = True
                    tool_call = event.tool_call
                    
                    logger.info(
                        "Executing MCP tool",
                        tool_name=tool_call.name,
                        iteration=iteration
                    )
                    
                    yield f"\n\n🔧 Calling {tool_call.name}...\n"
                    
                    # Execute the MCP tool
                    result = await self._execute_mcp_tool(
                        tool_call.name,
                        tool_call.arguments
                    )
                    
                    yield f"✅ Result: {result[:100]}{'...' if len(result) > 100 else ''}\n\n"
                    
                    # Add tool call and result to history
                    self.history.add_message(
                        LFAgentChatMessage(
                            role="assistant",
                            content=f"[Called tool: {tool_call.name}]"
                        )
                    )
                    self.history.add_message(
                        LFAgentChatMessage(
                            role="tool",
                            content=f"Tool result: {result}"
                        )
                    )
                    
                    # Prepare for next iteration
                    current_input = None  # History already updated
                    break  # Exit event loop, continue while loop
            
            # If no tool was called, we're done
            if not tool_call_made:
                logger.info("No tool call made, conversation complete")
                break
        
        # Save history
        if iteration >= self._max_iterations:
            logger.warning("Max iterations reached", max_iterations=self._max_iterations)
            yield "\n\n⚠️ Maximum tool calls reached.\n"
        
        self._persist_history()
    
    async def _execute_mcp_tool(
        self, tool_name: str, arguments: dict
    ) -> str:
        """Execute an MCP tool and return the result.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool parameters
        
        Returns:
            Tool result as string
        """
        # Find the tool class
        tool_class = next(
            (t for t in self._mcp_tools 
             if getattr(t, "mcp_tool_name", None) == tool_name),
            None
        )
        
        if not tool_class:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg, available_tools=[
                getattr(t, "mcp_tool_name", t.__name__) for t in self._mcp_tools
            ])
            return f"Error: {error_msg}"
        
        try:
            # Instantiate and execute tool
            tool_instance = tool_class()
            input_schema_class = tool_class.input_schema
            
            # Create input with tool_name discriminator
            tool_input = input_schema_class(
                tool_name=tool_name,
                **arguments
            )
            
            # Execute tool
            result = await tool_instance.arun(tool_input)
            
            # Extract result content
            result_content = getattr(result, "result", str(result))
            
            logger.info(
                "Tool execution successful",
                tool_name=tool_name,
                result_length=len(str(result_content))
            )
            
            return str(result_content)
        
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
```

---

## Summary of Changes

### Files to Create
1. **`agents/llamagent/types.py`** - NEW
   - `ToolDefinition` - Provider-agnostic tool format
   - `ToolCallRequest` - Standardized tool call request
   - `StreamEvent` - Unified streaming event format

### Files to Modify
2. **`agents/llamagent/clients/client.py`**
   - Replace `chat()` and `stream_chat()` with `stream_chat_with_tools()`
   - Single method that handles everything

3. **`agents/llamagent/clients/openai.py`**
   - Implement `stream_chat_with_tools()` using native function calling
   - Convert `ToolDefinition` → OpenAI format
   - Parse OpenAI `tool_calls` → `StreamEvent`

4. **`agents/llamagent/clients/ollama.py`**
   - Implement `stream_chat_with_tools()` using JSON prompting
   - Inject tools into system prompt
   - Detect JSON → `StreamEvent`

5. **`agents/llamagent/agent.py`**
   - Add `stream_chat_with_tools()` method
   - Delegates to client, manages history
   - Provider-agnostic interface

6. **`agents/chat_orchestrator.py`**
   - Simplify to use `LFAgent.stream_chat_with_tools()`
   - Never touches client directly
   - Handles MCP tool execution loop

### Files to Remove
7. **`context_providers/mcp_tools_context_provider.py`** - DELETE
   - No longer needed
   - Tools handled by client implementations

---

## Benefits

✅ **Clean abstraction** - Each layer has clear responsibility  
✅ **Consistent interface** - All clients return same format  
✅ **Provider-agnostic** - Orchestrator doesn't know about clients  
✅ **No backwards compatibility** - Optimal design  
✅ **Easy to test** - Can mock `StreamEvent` responses  
✅ **Easy to extend** - New provider = implement client interface  

---

## Implementation Order

### Phase 1: Types & Base (0.5 day)
- Create `types.py` with all types
- Update `LFAgentClient` abstract interface
- Update `LFAgent` with `stream_chat_with_tools()`

### Phase 2: OpenAI Client (1 day)
- Implement native function calling
- Test with OpenAI and Lemonade

### Phase 3: Ollama Client (1 day)
- Implement JSON-based tool calling
- Test with Ollama

### Phase 4: Orchestrator (1 day)
- Simplify to use `LFAgent` interface
- Remove direct client access
- Test full flow

### Phase 5: Cleanup & Testing (0.5 day)
- Delete old context provider
- Integration tests
- Documentation

**Total: 4 days**

