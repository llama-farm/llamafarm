"""
Atmosphere Adapter System

Adapters connect Atmosphere to external AI backends (LlamaFarm, Ollama, vLLM, etc.)

Adding a new adapter:
1. Create a file in this directory (e.g., my_backend.py)
2. Implement the Adapter base class
3. Register with @register_adapter decorator
4. Done! It will be auto-discovered.

Example:
    # adapters/my_backend.py
    from . import Adapter, Capability, register_adapter
    
    @register_adapter("my-backend")
    class MyBackendAdapter(Adapter):
        DEFAULT_URL = "http://localhost:8000"
        
        async def connect(self) -> bool:
            # Connect to your backend
            return True
        
        async def get_capabilities(self) -> List[Capability]:
            return [Capability(name="inference", description="LLM inference")]
        
        async def execute(self, capability: str, **kwargs) -> Any:
            # Execute the capability
            pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
import logging
import importlib
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """A capability provided by an adapter."""
    name: str
    description: str
    models: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # For semantic routing
    vector: Optional[List[float]] = None


@dataclass 
class AdapterInfo:
    """Metadata about a registered adapter."""
    name: str
    cls: Type["Adapter"]
    description: str = ""
    priority: int = 0  # Higher = preferred when multiple can handle same capability


class Adapter(ABC):
    """
    Base class for all Atmosphere adapters.
    
    An adapter connects Atmosphere to an external AI backend
    (LlamaFarm, Ollama, vLLM, OpenAI, Anthropic, etc.)
    
    Implement this class to add support for a new backend.
    """
    
    # Override in subclass
    NAME: str = "base"
    DESCRIPTION: str = "Base adapter"
    DEFAULT_URL: str = "http://localhost:8000"
    PRIORITY: int = 0  # Higher = preferred
    
    def __init__(self, url: str = None, **config):
        self.url = (url or self.DEFAULT_URL).rstrip('/')
        self.config = config
        self._connected = False
        self._capabilities: List[Capability] = []
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def capabilities(self) -> List[Capability]:
        return self._capabilities
    
    # === Required Methods ===
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the backend.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from the backend."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if backend is healthy.
        
        Returns:
            True if backend is responding
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[Capability]:
        """
        Discover capabilities provided by this backend.
        
        Returns:
            List of Capability objects
        """
        pass
    
    @abstractmethod
    async def execute(self, capability: str, **kwargs) -> Any:
        """
        Execute a capability.
        
        Args:
            capability: Name of capability to execute
            **kwargs: Capability-specific arguments
        
        Returns:
            Result of execution
        """
        pass
    
    # === Optional Methods (override if supported) ===
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts. Override if backend supports embeddings."""
        raise NotImplementedError(f"{self.NAME} does not support embeddings")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text. Override if backend supports generation."""
        raise NotImplementedError(f"{self.NAME} does not support generation")
    
    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion. Override if backend supports chat."""
        raise NotImplementedError(f"{self.NAME} does not support chat")
    
    # === Helper Methods ===
    
    def to_dict(self) -> Dict:
        """Serialize adapter state."""
        return {
            "name": self.NAME,
            "url": self.url,
            "connected": self._connected,
            "capabilities": [
                {"name": c.name, "description": c.description, "models": c.models}
                for c in self._capabilities
            ]
        }


# === Adapter Registry ===

_adapters: Dict[str, AdapterInfo] = {}


def register_adapter(name: str = None, description: str = "", priority: int = 0):
    """
    Decorator to register an adapter.
    
    Usage:
        @register_adapter("my-backend", description="My AI backend", priority=5)
        class MyAdapter(Adapter):
            ...
    """
    def decorator(cls: Type[Adapter]) -> Type[Adapter]:
        adapter_name = name or cls.NAME
        _adapters[adapter_name] = AdapterInfo(
            name=adapter_name,
            cls=cls,
            description=description or cls.DESCRIPTION,
            priority=priority or cls.PRIORITY
        )
        logger.debug(f"Registered adapter: {adapter_name}")
        return cls
    return decorator


def get_adapter(name: str) -> Optional[Type[Adapter]]:
    """Get an adapter class by name."""
    info = _adapters.get(name)
    return info.cls if info else None


def list_adapters() -> List[AdapterInfo]:
    """List all registered adapters."""
    return sorted(_adapters.values(), key=lambda a: -a.priority)


def create_adapter(name: str, url: str = None, **config) -> Optional[Adapter]:
    """Create an adapter instance by name."""
    cls = get_adapter(name)
    if cls:
        return cls(url=url, **config)
    return None


# === Auto-Discovery ===

def discover_adapters():
    """
    Auto-discover and load all adapters in this package.
    
    Scans for Python files in the adapters directory and imports them,
    which triggers their @register_adapter decorators.
    """
    package_dir = Path(__file__).parent
    
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        if module_info.name.startswith('_'):
            continue
        
        try:
            importlib.import_module(f".{module_info.name}", package=__name__)
            logger.debug(f"Loaded adapter module: {module_info.name}")
        except Exception as e:
            logger.warning(f"Failed to load adapter {module_info.name}: {e}")


# === Adapter Manager ===

class AdapterManager:
    """
    Manages all adapters and provides unified access to capabilities.
    
    Usage:
        manager = AdapterManager()
        await manager.discover_all()
        
        # Find adapter for a capability
        adapter = manager.find_adapter_for("embeddings")
        result = await adapter.embed(["hello world"])
        
        # Or use unified interface
        result = await manager.execute("embeddings", texts=["hello"])
    """
    
    def __init__(self):
        self._instances: Dict[str, Adapter] = {}
        self._capability_map: Dict[str, List[str]] = {}  # capability -> [adapter_names]
    
    async def discover_all(self, urls: Dict[str, str] = None):
        """
        Discover and connect to all available backends.
        
        Args:
            urls: Optional dict of adapter_name -> url overrides
        """
        urls = urls or {}
        
        # Ensure adapters are loaded
        discover_adapters()
        
        for info in list_adapters():
            url = urls.get(info.name)
            adapter = info.cls(url=url)
            
            try:
                if await adapter.connect():
                    self._instances[info.name] = adapter
                    
                    # Map capabilities
                    caps = await adapter.get_capabilities()
                    for cap in caps:
                        if cap.name not in self._capability_map:
                            self._capability_map[cap.name] = []
                        self._capability_map[cap.name].append(info.name)
                    
                    logger.info(f"Connected to {info.name} with {len(caps)} capabilities")
            except Exception as e:
                logger.debug(f"Could not connect to {info.name}: {e}")
    
    async def disconnect_all(self):
        """Disconnect from all backends."""
        for adapter in self._instances.values():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {adapter.NAME}: {e}")
        self._instances.clear()
        self._capability_map.clear()
    
    def find_adapter_for(self, capability: str) -> Optional[Adapter]:
        """Find the best adapter for a capability."""
        adapter_names = self._capability_map.get(capability, [])
        
        if not adapter_names:
            return None
        
        # Return highest priority connected adapter
        for name in adapter_names:
            adapter = self._instances.get(name)
            if adapter and adapter.is_connected:
                return adapter
        
        return None
    
    async def execute(self, capability: str, **kwargs) -> Any:
        """Execute a capability using the best available adapter."""
        adapter = self.find_adapter_for(capability)
        if not adapter:
            raise RuntimeError(f"No adapter available for capability: {capability}")
        
        return await adapter.execute(capability, **kwargs)
    
    def get_all_capabilities(self) -> List[Capability]:
        """Get all capabilities from all connected adapters."""
        caps = []
        for adapter in self._instances.values():
            caps.extend(adapter.capabilities)
        return caps
    
    @property
    def connected_adapters(self) -> List[str]:
        """List names of connected adapters."""
        return list(self._instances.keys())
    
    def get_status(self) -> Dict:
        """Get status of all adapters."""
        return {
            "adapters": {
                name: adapter.to_dict()
                for name, adapter in self._instances.items()
            },
            "capabilities": dict(self._capability_map)
        }


# Export public API
__all__ = [
    # Base classes
    "Adapter",
    "Capability",
    "AdapterInfo",
    
    # Registry functions
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "create_adapter",
    "discover_adapters",
    
    # Manager
    "AdapterManager",
]

# Auto-discover on import
discover_adapters()
