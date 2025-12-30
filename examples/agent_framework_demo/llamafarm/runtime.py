"""LlamaFarm 'Magic' Runtime.

This script acts as the bridge between LlamaFarm and user code.
It leverages the MCP protocol to expose user-defined @tools
and manage Active Agents without requiring the user to write any server code.
"""

import sys
import importlib.util
import asyncio
import inspect
from pathlib import Path
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

# Import our local SDK to check for registered tools
from llamafarm import sdk

def load_user_module(path: str) -> Any:
    """Dynamically load the user's python script."""
    file_path = Path(path).resolve()
    if not file_path.exists():
        print(f"Error: User file not found: {file_path}", file=sys.stderr)
        sys.exit(1)
        
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        print(f"Error: Could not load module spec for {file_path}", file=sys.stderr)
        sys.exit(1)
        
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error executing user module: {e}", file=sys.stderr)
        sys.exit(1)
        
    return module

def discover_agents(module: Any) -> list[sdk.Agent]:
    """Discover Agent subclasses in the module and instantiate them."""
    agents = []
    print("Scanning for Agents...", file=sys.stderr)
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, sdk.Agent) and 
            obj is not sdk.Agent):
            print(f"Found Agent: {name}", file=sys.stderr)
            # Instantiate the agent
            try:
                agent = obj()
                agents.append(agent)
            except Exception as e:
                print(f"Error instantiating {name}: {e}", file=sys.stderr)
    return agents

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m llamafarm.runtime <path_to_user_script.py>", file=sys.stderr)
        sys.exit(1)
        
    user_script = sys.argv[1]
    
    # 1. Load User Code first to discover what we need
    print(f"Loading user code from {user_script}...", file=sys.stderr)
    user_module = load_user_module(user_script)
    
    # 2. Discover Agents
    agents = discover_agents(user_module)
    if agents:
        print(f"Discovered {len(agents)} active agents.", file=sys.stderr)
    
    # 3. Define Lifespan to run Agents
    @asynccontextmanager
    async def agent_lifespan(server: FastMCP) -> AsyncIterator[Any]:
        # Initialize Client
        # In a real app, base_url would come from config/env
        client = sdk.LlamaFarmClient("http://127.0.0.1:11540")
        
        # Start all agents
        tasks = []
        for agent in agents:
            print(f"Starting agent: {agent.name}", file=sys.stderr)
            # Inject client
            agent.set_client(client)
            task = asyncio.create_task(agent.start())
            tasks.append(task)
            
        try:
            yield 
        finally:
            # Stop all agents
            print("Stopping agents...", file=sys.stderr)
            for agent in agents:
                await agent.stop()
                
            # Cancel tasks
            for task in tasks:
                task.cancel()
            
            # Close client
            await client.close()
    
    # 4. Initialize FastMCP Server with Lifespan
    server_name = Path(user_script).stem
    mcp = FastMCP(server_name, lifespan=agent_lifespan if agents else None)
    
    # 5. Discover and Register Tools from SDK Registry
    count = 0
    for name, func in sdk._TOOLS_REGISTRY.items():
        print(f"Registering tool: {name}", file=sys.stderr)
        mcp.add_tool(func)
        count += 1
        
    if count == 0:
        print("Warning: No @tool functions found.", file=sys.stderr)
    else:
        print(f"Successfully registered {count} tools.", file=sys.stderr)

    # 6. Run the server
    print("Starting Magic Runtime...", file=sys.stderr)
    mcp.run()

if __name__ == "__main__":
    main()
