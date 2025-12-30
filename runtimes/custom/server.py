import asyncio
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import BaseModel

# Import config loading
from config import load_config

# Initialize MCP Server
app = Server("custom-runtime")

# Global state
LOADED_FILES = []
REGISTERED_TOOLS = {}

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

def import_source_file(file_path: str):
    """Dynamically import a source file"""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        LOADED_FILES.append(file_path)
        
        # Scan for tools
        for name, obj in vars(module).items():
            if callable(obj) and getattr(obj, "_is_tool", False):
                tool_name = getattr(obj, "_tool_name", name)
                
                # Register tool with MCP Server
                @app.tool(name=tool_name, description=getattr(obj, "_tool_description", None))
                def wrapper(*args, _func=obj, **kwargs):
                    return _func(*args, **kwargs)
                
                REGISTERED_TOOLS[tool_name] = obj
                print(f"Registered tool: {tool_name}", file=sys.stderr)

async def main():
    print("Starting Custom Runtime (Stdio)...", file=sys.stderr)
    
    # Load config
    config_path = os.environ.get("LF_CONFIG_PATH", "llamafarm.yaml")
    
    # Find config path logic
    if not os.path.exists(config_path):
        # Scan up directories
        curr = Path.cwd()
        for _ in range(5): # Max depth
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
                    
                    # Convert relative path to absolute
                    abs_path = project_root / path
                    
                    # Add directory to sys.path to support local imports
                    sys.path.insert(0, str(abs_path.parent))
                    
                    import_source_file(str(abs_path))
                    
        except Exception as e:
             print(f"Failed to load custom code: {e}", file=sys.stderr)

    # Run the stdio server
    from mcp.server import InitializationOptions, NotificationOptions
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream, 
            write_stream, 
            InitializationOptions(
                server_name="custom-runtime",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
