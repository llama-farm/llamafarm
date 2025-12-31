import httpx
import asyncio
from typing import Any, Callable, Optional, Dict, List, Union
from functools import wraps
import inspect

# Global registry of tools
_TOOLS_REGISTRY = {}

def tool(name_or_func: Union[str, Callable, None] = None, *, name: Optional[str] = None):
    """
    Decorator to mark a function as a LlamaFarm tool.
    
    The description is ALWAYS taken from the function's docstring.
    
    Usage:
    @tool
    def my_func(): 
        '''This is the description.'''
        ...
    
    @tool(name="custom_name")
    def my_func(): ...
    """
    
    # CASE 1: Used as @tool (no parens)
    if callable(name_or_func):
        func = name_or_func
        return _create_tool_wrapper(func, name=None)

    # CASE 2: Used as @tool(...) with arguments
    actual_name = name or (name_or_func if isinstance(name_or_func, str) else None)
    
    def decorator(func: Callable):
        return _create_tool_wrapper(func, name=actual_name)
        
    return decorator

def _create_tool_wrapper(func: Callable, name: Optional[str]):
    tool_name = name or func.__name__
    # Enforce using docstring for description
    tool_description = func.__doc__.strip() if func.__doc__ else "No description provided"
    
    func._is_tool = True
    func._tool_name = tool_name
    func._tool_description = tool_description
    _TOOLS_REGISTRY[tool_name] = func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
        
    # Copy attributes
    wrapper._is_tool = True
    wrapper._tool_name = tool_name
    wrapper._tool_description = tool_description
    return wrapper

class LlamaFarmClient:
    """Simple client for LlamaFarm Universal Runtime."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11540"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        
    async def post(self, path: str, json: Any) -> Any:
        resp = await self.client.post(path, json=json)
        resp.raise_for_status()
        return resp.json()
        
    async def get(self, path: str) -> Any:
        resp = await self.client.get(path)
        resp.raise_for_status()
        return resp.json()
        
    async def close(self):
        await self.client.aclose()

class Agent:
    """Base class for Universal LlamaFarm Agents."""
    
    interval: float = 1.0
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._is_running = False
        self.client: Optional[LlamaFarmClient] = None
        
    def set_client(self, client: LlamaFarmClient):
        self.client = client
        
    async def on_start(self):
        """Called when the agent starts. Override this."""
        pass
        
    async def on_stop(self):
        """Called when the agent stops. Override this."""
        pass
        
    async def on_tick(self):
        """Called every interval. Override this."""
        pass
        
    async def start(self):
        if self._is_running: return
        self._is_running = True
        print(f"[{self.name}] Starting...")
        
        try:
            await self.on_start()
            while self._is_running:
                await self.on_tick()
                await asyncio.sleep(self.interval)
        except Exception as e:
            print(f"[{self.name}] Crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.stop()
            
    async def stop(self):
        self._is_running = False
        await self.on_stop()
        print(f"[{self.name}] Stopped.")
