"""
LlamaFarm Adapter for Atmosphere.

Connects to LlamaFarm server for:
- Text embeddings
- LLM inference  
- RAG queries
- Model management
"""

import logging
from typing import Any, Dict, List, Optional
import aiohttp

from . import Adapter, Capability, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("llamafarm", description="LlamaFarm AI server", priority=10)
class LlamaFarmAdapter(Adapter):
    """
    Adapter for LlamaFarm server.
    
    LlamaFarm provides unified access to local LLMs, embeddings, and RAG.
    """
    
    NAME = "llamafarm"
    DESCRIPTION = "LlamaFarm AI server"
    DEFAULT_URL = "http://localhost:14345"
    PRIORITY = 10  # Prefer LlamaFarm over Ollama
    
    def __init__(self, url: str = None, **config):
        super().__init__(url, **config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: List[str] = []
    
    async def connect(self) -> bool:
        """Connect to LlamaFarm server."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            if not await self.health_check():
                await self._session.close()
                self._session = None
                return False
            
            # Discover capabilities
            self._capabilities = await self.get_capabilities()
            self._connected = True
            
            logger.info(f"Connected to LlamaFarm at {self.url}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to LlamaFarm: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False
    
    async def disconnect(self):
        """Disconnect from LlamaFarm."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
    
    async def health_check(self) -> bool:
        """Check LlamaFarm health."""
        try:
            async with self._session.get(
                f"{self.url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def get_capabilities(self) -> List[Capability]:
        """Discover LlamaFarm capabilities."""
        caps = []
        
        # Always has embeddings
        caps.append(Capability(
            name="embeddings",
            description="Text embeddings via sentence-transformers",
            models=["sentence-transformers/all-MiniLM-L6-v2"]
        ))
        
        # Check for models
        try:
            async with self._session.get(f"{self.url}/v1/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._models = [
                        m.get("id", m.get("name", "unknown"))
                        for m in data.get("data", [])
                    ]
                    if self._models:
                        caps.append(Capability(
                            name="llm-inference",
                            description="LLM text generation",
                            models=self._models
                        ))
                        caps.append(Capability(
                            name="chat",
                            description="Chat completion",
                            models=self._models
                        ))
        except Exception as e:
            logger.debug(f"Could not list models: {e}")
        
        # Check for RAG
        try:
            async with self._session.get(f"{self.url}/v1/rag/status") as resp:
                if resp.status == 200:
                    caps.append(Capability(
                        name="rag",
                        description="RAG knowledge retrieval"
                    ))
        except Exception:
            pass
        
        return caps
    
    async def execute(self, capability: str, **kwargs) -> Any:
        """Execute a capability."""
        if capability == "embeddings":
            texts = kwargs.get("texts") or [kwargs.get("text", "")]
            return await self.embed(texts)
        
        elif capability in ("llm-inference", "generate"):
            prompt = kwargs.get("prompt", "")
            return await self.generate(prompt, **kwargs)
        
        elif capability == "chat":
            messages = kwargs.get("messages", [])
            return await self.chat(messages, **kwargs)
        
        elif capability == "rag":
            query = kwargs.get("query", "")
            return await self.rag_query(query, **kwargs)
        
        else:
            raise ValueError(f"Unknown capability: {capability}")
    
    # === Embedding API ===
    
    async def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for texts."""
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
    
    # === Generation API ===
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        if kwargs.get("model"):
            payload["model"] = kwargs["model"]
        
        async with self._session.post(
            f"{self.url}/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Generation failed: {resp.status} - {error}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    
    # === Chat API ===
    
    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        payload = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
        }
        if kwargs.get("model"):
            payload["model"] = kwargs["model"]
        
        async with self._session.post(
            f"{self.url}/v1/chat/completions",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Chat failed: {resp.status} - {error}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]
    
    # === RAG API ===
    
    async def rag_query(self, query: str, **kwargs) -> List[Dict]:
        """Query RAG knowledge base."""
        if not self._connected:
            raise RuntimeError("Not connected to LlamaFarm")
        
        payload = {
            "query": query,
            "top_k": kwargs.get("top_k", 5)
        }
        if kwargs.get("collection"):
            payload["collection"] = kwargs["collection"]
        
        async with self._session.post(
            f"{self.url}/v1/rag/query",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"RAG query failed: {resp.status} - {error}")
            return await resp.json()
