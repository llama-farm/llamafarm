import asyncio
import importlib.util
import os
import subprocess
import sys
import inspect
from pathlib import Path
from typing import List, Any, AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context
from mcp.server import Server # Fallback if needed, but we used FastMCP in demo

# Import local SDK from same directory
import sdk

# Global state
LOADED_FILES = []
REGISTERED_AGENTS: List[sdk.Agent] = []

def install_dependencies(deps: List[str]):
    """Install dependencies using uv"""
    if not deps:
        return
    
    # Use stderr for logging so it doesn't interfere with stdio communication
    print(f"Installing dependencies: {', '.join(deps)}", file=sys.stderr)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "uv", "pip", "install"] + deps,
            stdout=sys.stderr,
            stderr=sys.stderr
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}", file=sys.stderr)

def import_source_file(file_path: str, mcp: FastMCP):
    """Dynamically import a source file and register tools/agents"""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            LOADED_FILES.append(file_path)
            
            # Scan for tools
            tools = discover_tools(module)
            for tool_name, tool_func in tools.items():
                tool_description = getattr(tool_func, "_tool_description", None)
                print(f"Registering tool: {tool_name}", file=sys.stderr)
                mcp.add_tool(tool_func, name=tool_name, description=tool_description)
            
            # Scan for Agents
            agents = discover_agents(module)
            for agent_cls in agents:
                print(f"Found Agent: {agent_cls.__name__}", file=sys.stderr)
                try:
                    agent = agent_cls()
                    REGISTERED_AGENTS.append(agent)
                except Exception as e:
                    print(f"Error instantiating {agent_cls.__name__}: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Error loading module {module_name}: {e}", file=sys.stderr)

def discover_tools(module: Any) -> dict:
    """Find all @tool decorated functions in a module."""
    tools = {}
    for name, obj in vars(module).items():
        if callable(obj) and getattr(obj, "_is_tool", False):
            tool_name = getattr(obj, "_tool_name", name)
            tools[tool_name] = obj
    return tools

def discover_agents(module: Any) -> list:
    """Find all Agent subclasses in a module."""
    agents = []
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and 
            issubclass(obj, sdk.Agent) and 
            obj is not sdk.Agent):
            agents.append(obj)
    return agents

@asynccontextmanager
async def agent_lifespan(server: FastMCP) -> AsyncIterator[Any]:
    """Lifespan manager for running background agents."""
    
    # Initialize Client (Universal Runtime usually on 11540)
    # TODO: Get URL from config/env
    client = sdk.LlamaFarmClient("http://127.0.0.1:11540")
    
    if REGISTERED_AGENTS:
        print(f"Starting {len(REGISTERED_AGENTS)} agents...", file=sys.stderr)
    
    tasks = []
    for agent in REGISTERED_AGENTS:
        print(f"Starting agent: {agent.name}", file=sys.stderr)
        agent.set_client(client)
        task = asyncio.create_task(agent.start())
        tasks.append(task)
        
    try:
        yield 
    finally:
        # Stop all agents
        if REGISTERED_AGENTS:
            print("Stopping agents...", file=sys.stderr)
        
        for agent in REGISTERED_AGENTS:
            try:
                await agent.stop()
            except Exception as e:
                print(f"Error stopping agent {agent.name}: {e}", file=sys.stderr)
            
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        await client.close()

def main():
    print("Starting Custom Runtime (Stdio)...", file=sys.stderr)
    
    # Initialize FastMCP Server
    # We pass dependencies/tools later
    mcp = FastMCP("custom-runtime", lifespan=agent_lifespan)
    
    # Import config loading
    try:
        from config import load_config
        has_config_lib = True
    except ImportError:
        # Fallback if running outside of monorepo context without paths set?
        # But we expect to be run via 'uv run' in the right env
        has_config_lib = False
        print("Warning: Could not import 'config' library.", file=sys.stderr)
    
    if has_config_lib:
        # Load config logic
        config_path = os.environ.get("LF_CONFIG_PATH", "llamafarm.yaml")
        
        # Find config path logic
        if not os.path.exists(config_path):
            curr = Path.cwd()
            for _ in range(5): 
                p = curr / "llamafarm.yaml"
                if p.exists():
                    config_path = str(p)
                    break
                curr = curr.parent

        if os.path.exists(config_path):
            try:
                print(f"Loading config from {config_path}", file=sys.stderr)
                config = load_config(config_path=config_path)
                     
                if config.custom_code:
                    project_root = Path(config_path).parent
                    for code_item in config.custom_code:
                        path = code_item.path
                        if code_item.dependencies:
                            install_dependencies(code_item.dependencies)
                        
                        abs_path = project_root / path
                        
                        # Add directory to sys.path
                        if str(abs_path.parent) not in sys.path:
                            sys.path.insert(0, str(abs_path.parent))
                        
                        import_source_file(str(abs_path), mcp)
                        
            except Exception as e:
                 print(f"Failed to load custom code: {e}", file=sys.stderr)

    # Run server
    mcp.run()

if __name__ == "__main__":
    main()
