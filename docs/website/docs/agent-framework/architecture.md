# Architecture: The Magic Runtime

The core of the Agent Framework is the **Magic Runtime** (`llamafarm.runtime`).

## How it Works

1.  **Dynamic Loading**: The runtime accepts a path to a user Python file (e.g., `src/my_logic.py`). It uses `importlib` to load this file as a module.
2.  **Discovery**:
    *   **Tools**: It inspects `llamafarm.sdk._TOOLS_REGISTRY` to find functions decorated with `@tool`.
    *   **Agents**: It inspects the module members to find subclasses of `llamafarm.sdk.Agent`.
3.  **Concurrency**:
    *   It initializes a `FastMCP` server (based on Starlette/FastAPI) to serve the Tools over the MCP protocol.
    *   It creates `asyncio.Task`s for each discovered Agent.
    *   It runs the MCP Server and Agent Tasks in a shared AsyncIO Event Loop.
4.  **Client Injection**:
    *   It instantiates a `LlamaFarmClient` and injects it into every Agent instance, enabling immediate connectivity to the rest of the LlamaFarm platform.

## Diagram

```mermaid
graph TD
    UserCode[User Script] -->|Defines| Tools[@tool]
    UserCode -->|Defines| Agents[Agent Classes]
    
    Runtime[Magic Runtime] -->|Loads| UserCode
    Runtime -->|Serves| MCPServer[MCP Server]
    Runtime -->|Runs| AgentLoop[Agent Loops]
    
    MCPServer -->|Exposes| Tools
    AgentLoop -->|Executes| Agents
    
    Agents -->|Calls| Client[LlamaFarm Client]
    Client -->|HTTP| UniversalRuntime[Universal Runtime]
```
