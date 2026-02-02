"""
LlamaFarm Adapter for Atmosphere.

Discovers and integrates with local LlamaFarm instance.
Provides capabilities: embeddings, inference, RAG.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import aiohttp

logger = logging.getLogger(__name__)


# Compatibility classes for existing Atmosphere code
@dataclass
class LlamaFarmConfig:
    """Configuration for LlamaFarm connection."""
    host: str = "localhost"
    port: int = 14345
    api_key: Optional[str] = None


class LlamaFarmBackend:
    """
    Legacy compatibility class for LlamaFarm backend.
    
    Wraps LlamaFarmAdapter for backwards compatibility.
    """
    
    def __init__(self, config: LlamaFarmConfig = None):
        self.config = config or LlamaFarmConfig()
        self._adapter: Optional["LlamaFarmAdapter"] = None
    
    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"
    
    async def connect(self) -> bool:
        self._adapter = LlamaFarmAdapter(self.base_url)
        return await self._adapter.connect()
    
    async def disconnect(self):
        if self._adapter:
            await self._adapter.disconnect()
    
    async def embed(self, text: str) -> List[float]:
        if not self._adapter:
            raise RuntimeError("Not connected")
        return await self._adapter.embed_single(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._adapter:
            raise RuntimeError("Not connected")
        return await self._adapter.embed(texts)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self._adapter:
            raise RuntimeError("Not connected")
        return await self._adapter.generate(prompt, **kwargs)


@dataclass
class LlamaFarmCapability:
    """A capability provided by LlamaFarm."""
    name: str
    description: str
    endpoint: str
    models: List[str] = field(default_factory=list)


class LlamaFarmAdapter:
    """
    Connect to local or remote LlamaFarm instance.
    
    Auto-discovers capabilities and registers them with the mesh.
    """
    
    DEFAULT_URL = "http://localhost:14345"
    DISCOVER_TIMEOUT = 5.0
    
    def __init__(self, url: str = DEFAULT_URL):
        self.url = url.rstrip('/')
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self.capabilities: List[LlamaFarmCapability] = []
        self.models: List[str] = []
    
    async def connect(self) -> bool:
        """Connect to LlamaFarm and discover capabilities."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Check health
            async with self._session.get(
                f"{self.url}/health",
                timeout=aiohttp.ClientTimeout(total=self.DISCOVER_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"LlamaFarm health check failed: {resp.status}")
                    return False
                health = await resp.json()
                logger.info(f"LlamaFarm connected: {health}")
            
            # Discover models
            try:
                async with self._session.get(f"{self.url}/v1/models") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.models = [m.get("id", m.get("name", "unknown")) 
                                      for m in data.get("data", [])]
            except Exception as e:
                logger.debug(f"Could not list models: {e}")
            
            # Build capability list
            self.capabilities = [
                LlamaFarmCapability(
                    name="embeddings",
                    description="Text embeddings via sentence-transformers",
                    endpoint="/v1/nlp/embeddings",
                    models=["sentence-transformers/all-MiniLM-L6-v2"]
                ),
                LlamaFarmCapability(
                    name="llm-inference",
                    description="LLM text generation",
                    endpoint="/v1/chat/completions",
                    models=self.models
                ),
            ]
            
            # Check for RAG capability
            try:
                async with self._session.get(f"{self.url}/v1/rag/status") as resp:
                    if resp.status == 200:
                        self.capabilities.append(LlamaFarmCapability(
                            name="rag",
                            description="RAG knowledge retrieval",
                            endpoint="/v1/rag/query",
                            models=[]
                        ))
            except Exception:
                pass
            
            self._connected = True
            logger.info(f"LlamaFarm adapter connected with {len(self.capabilities)} capabilities")
            return True
            
        except aiohttp.ClientError as e:
            logger.warning(f"Could not connect to LlamaFarm at {self.url}: {e}")
            return False
        except Exception as e:
            logger.error(f"LlamaFarm discovery error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from LlamaFarm."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # === Embedding API ===
    
    async def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Get embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use (default: all-MiniLM-L6-v2)
        
        Returns:
            List of embedding vectors (384-dim by default)
        """
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        model = model or "sentence-transformers/all-MiniLM-L6-v2"
        
        async with self._session.post(
            f"{self.url}/v1/nlp/embeddings",
            json={"input": texts, "model": model}
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Embedding failed: {resp.status}")
            data = await resp.json()
            return [item["embedding"] for item in data["data"]]
    
    async def embed_single(self, text: str, model: str = None) -> List[float]:
        """Embed single text."""
        results = await self.embed([text], model)
        return results[0]
    
    # === Inference API ===
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        if model:
            payload["model"] = model
        
        async with self._session.post(
            f"{self.url}/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Generation failed: {resp.status} - {error}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    
    # === RAG API ===
    
    async def rag_query(
        self,
        query: str,
        collection: str = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query RAG knowledge base.
        
        Args:
            query: Search query
            collection: Collection to search (optional)
            top_k: Number of results
        
        Returns:
            List of relevant documents with scores
        """
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        payload = {"query": query, "top_k": top_k}
        if collection:
            payload["collection"] = collection
        
        async with self._session.post(
            f"{self.url}/v1/rag/query",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"RAG query failed: {resp.status} - {error}")
            return await resp.json()
    
    # === Capability Registration ===
    
    def get_capability_vectors(self) -> Dict[str, List[float]]:
        """
        Get capability description vectors for gossip.
        
        Called by gossip protocol to register this node's capabilities.
        """
        # These would be pre-computed embeddings of capability descriptions
        # For now, return placeholder (real impl would call self.embed())
        return {
            cap.name: []  # Will be filled by gossip protocol using embed()
            for cap in self.capabilities
        }
    
    def to_gossip_info(self) -> List[Dict]:
        """Convert capabilities to gossip-ready format."""
        return [
            {
                "id": f"llamafarm:{cap.name}",
                "label": cap.name,
                "description": cap.description,
                "models": cap.models,
                "endpoint": cap.endpoint,
                "local": True,
            }
            for cap in self.capabilities
        ]


# Module-level singleton for easy access
_adapter: Optional[LlamaFarmAdapter] = None


async def get_llamafarm_adapter(url: str = None) -> LlamaFarmAdapter:
    """Get or create LlamaFarm adapter."""
    global _adapter
    
    if _adapter is None:
        _adapter = LlamaFarmAdapter(url or LlamaFarmAdapter.DEFAULT_URL)
        await _adapter.connect()
    
    return _adapter


async def discover_llamafarm() -> Optional[LlamaFarmAdapter]:
    """
    Discover LlamaFarm on common ports.
    
    Tries:
    - localhost:14345 (LlamaFarm default)
    - localhost:11434 (Ollama, if LlamaFarm not found)
    """
    # Try LlamaFarm first
    adapter = LlamaFarmAdapter("http://localhost:14345")
    if await adapter.connect():
        return adapter
    
    # Could add Ollama fallback here
    
    return None
