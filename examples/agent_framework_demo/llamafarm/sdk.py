import httpx
import asyncio
from typing import Any, Callable, Optional, Dict, List
from functools import wraps

# Global registry of tools
_TOOLS_REGISTRY = {}

def tool(name: Optional[str] = None):
    """Decorator to mark a function as a LlamaFarm tool."""
    def decorator(func: Callable):
        tool_name = name or func.__name__
        func._is_tool = True
        func._tool_name = tool_name
        _TOOLS_REGISTRY[tool_name] = func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
        
    if callable(name):
        func = name
        name = None
        return decorator(func)
    return decorator

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
