"""
Ollama Adapter for Atmosphere.

Connects to local Ollama instance for:
- Text embeddings (nomic-embed-text)
- LLM inference
- Chat completion
"""

import logging
from typing import Any, Dict, List, Optional
import aiohttp

from . import Adapter, Capability, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("ollama", description="Ollama local LLM server", priority=5)
class OllamaAdapter(Adapter):
    """
    Adapter for Ollama server.
    
    Ollama provides easy local LLM serving. Lower priority than LlamaFarm
    since LlamaFarm has more features (RAG, projects, etc.)
    """
    
    NAME = "ollama"
    DESCRIPTION = "Ollama local LLM server"
    DEFAULT_URL = "http://localhost:11434"
    PRIORITY = 5  # Lower than LlamaFarm
    
    # Default embedding model
    EMBED_MODEL = "nomic-embed-text"
    
    def __init__(self, url: str = None, **config):
        super().__init__(url, **config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: List[str] = []
        self._embed_model = config.get("embed_model", self.EMBED_MODEL)
    
    async def connect(self) -> bool:
        """Connect to Ollama server."""
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
            
            logger.info(f"Connected to Ollama at {self.url}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False
    
    async def disconnect(self):
        """Disconnect from Ollama."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
    
    async def health_check(self) -> bool:
        """Check Ollama health."""
        try:
            async with self._session.get(
                f"{self.url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def get_capabilities(self) -> List[Capability]:
        """Discover Ollama capabilities."""
        caps = []
        
        try:
            async with self._session.get(f"{self.url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._models = [
                        m.get("name", "unknown")
                        for m in data.get("models", [])
                    ]
                    
                    # Check for embedding model
                    has_embed = any(
                        "embed" in m.lower() or "nomic" in m.lower()
                        for m in self._models
                    )
                    
                    if has_embed or self._embed_model:
                        caps.append(Capability(
                            name="embeddings",
                            description="Text embeddings via Ollama",
                            models=[self._embed_model]
                        ))
                    
                    # LLM models
                    llm_models = [
                        m for m in self._models
                        if "embed" not in m.lower()
                    ]
                    
                    if llm_models:
                        caps.append(Capability(
                            name="llm-inference",
                            description="LLM text generation via Ollama",
                            models=llm_models
                        ))
                        caps.append(Capability(
                            name="chat",
                            description="Chat completion via Ollama",
                            models=llm_models
                        ))
                        
        except Exception as e:
            logger.debug(f"Could not list models: {e}")
        
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
        
        else:
            raise ValueError(f"Unknown capability: {capability}")
    
    # === Embedding API ===
    
    async def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for texts."""
        if not self._connected:
            raise RuntimeError("Not connected to Ollama")
        
        model = model or self._embed_model
        
        results = []
        for text in texts:
            async with self._session.post(
                f"{self.url}/api/embeddings",
                json={"model": model, "prompt": text}
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Embedding failed: {resp.status}")
                data = await resp.json()
                results.append(data.get("embedding", []))
        
        return results
    
    # === Generation API ===
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        if not self._connected:
            raise RuntimeError("Not connected to Ollama")
        
        model = kwargs.get("model")
        if not model and self._models:
            # Use first non-embedding model
            model = next(
                (m for m in self._models if "embed" not in m.lower()),
                self._models[0]
            )
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        
        if kwargs.get("temperature"):
            payload["options"] = {"temperature": kwargs["temperature"]}
        
        async with self._session.post(
            f"{self.url}/api/generate",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Generation failed: {resp.status} - {error}")
            data = await resp.json()
            return data.get("response", "")
    
    # === Chat API ===
    
    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        if not self._connected:
            raise RuntimeError("Not connected to Ollama")
        
        model = kwargs.get("model")
        if not model and self._models:
            model = next(
                (m for m in self._models if "embed" not in m.lower()),
                self._models[0]
            )
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        
        if kwargs.get("temperature"):
            payload["options"] = {"temperature": kwargs["temperature"]}
        
        async with self._session.post(
            f"{self.url}/api/chat",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Chat failed: {resp.status} - {error}")
            data = await resp.json()
            return data.get("message", {}).get("content", "")
