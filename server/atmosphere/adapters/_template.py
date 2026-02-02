"""
Template Adapter for Atmosphere

Copy this file to create a new adapter:
1. cp _template.py my_backend.py
2. Replace MYBACKEND with your backend name
3. Implement the required methods
4. Done! It will be auto-discovered.

Example backends you might add:
- vLLM (high-performance local inference)
- OpenAI (cloud API)
- Anthropic (cloud API)
- HuggingFace (local or API)
- Replicate (cloud API)
- Together.ai (cloud API)
- Groq (fast inference)
- Mistral (cloud API)
- Cohere (embeddings + generation)
- Local GGUF/GGML models
"""

import logging
from typing import Any, Dict, List, Optional
import aiohttp

from . import Adapter, Capability, register_adapter

logger = logging.getLogger(__name__)


# Uncomment and modify the decorator when you're ready to register
# @register_adapter("my-backend", description="My AI backend", priority=5)
class TemplateAdapter(Adapter):
    """
    Template adapter - copy and customize this!
    
    Replace:
    - MYBACKEND with your backend name
    - DEFAULT_URL with your backend's URL
    - Implement the methods for your backend's API
    """
    
    NAME = "template"
    DESCRIPTION = "Template adapter - do not use directly"
    DEFAULT_URL = "http://localhost:8000"
    PRIORITY = 0
    
    def __init__(self, url: str = None, **config):
        super().__init__(url, **config)
        self._session: Optional[aiohttp.ClientSession] = None
        self._models: List[str] = []
        
        # Add any config specific to your backend
        # self._api_key = config.get("api_key")
    
    async def connect(self) -> bool:
        """Connect to your backend."""
        try:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
                # Add auth headers if needed:
                # headers={"Authorization": f"Bearer {self._api_key}"}
            )
            
            if not await self.health_check():
                await self._session.close()
                self._session = None
                return False
            
            self._capabilities = await self.get_capabilities()
            self._connected = True
            
            logger.info(f"Connected to {self.NAME} at {self.url}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to {self.NAME}: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False
    
    async def disconnect(self):
        """Disconnect from your backend."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
    
    async def health_check(self) -> bool:
        """Check if your backend is healthy."""
        try:
            # Customize the health check endpoint for your backend
            async with self._session.get(
                f"{self.url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False
    
    async def get_capabilities(self) -> List[Capability]:
        """Discover what your backend can do."""
        caps = []
        
        # Example: Add capabilities based on what your backend supports
        
        # If it supports embeddings:
        # caps.append(Capability(
        #     name="embeddings",
        #     description="Text embeddings",
        #     models=["your-embed-model"]
        # ))
        
        # If it supports generation:
        # caps.append(Capability(
        #     name="llm-inference",
        #     description="LLM text generation",
        #     models=self._models
        # ))
        
        # If it supports chat:
        # caps.append(Capability(
        #     name="chat",
        #     description="Chat completion",
        #     models=self._models
        # ))
        
        # If it has special capabilities:
        # caps.append(Capability(
        #     name="vision",
        #     description="Image analysis",
        #     models=["vision-model"]
        # ))
        
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
    
    # === Implement these methods for your backend ===
    
    async def embed(self, texts: List[str], model: str = None) -> List[List[float]]:
        """Get embeddings for texts."""
        if not self._connected:
            raise RuntimeError(f"Not connected to {self.NAME}")
        
        # Example API call (customize for your backend):
        # async with self._session.post(
        #     f"{self.url}/v1/embeddings",
        #     json={"input": texts, "model": model}
        # ) as resp:
        #     data = await resp.json()
        #     return [item["embedding"] for item in data["data"]]
        
        raise NotImplementedError("Implement embed() for your backend")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        if not self._connected:
            raise RuntimeError(f"Not connected to {self.NAME}")
        
        # Example API call (customize for your backend):
        # async with self._session.post(
        #     f"{self.url}/v1/completions",
        #     json={"prompt": prompt, "max_tokens": kwargs.get("max_tokens", 1000)}
        # ) as resp:
        #     data = await resp.json()
        #     return data["choices"][0]["text"]
        
        raise NotImplementedError("Implement generate() for your backend")
    
    async def chat(self, messages: List[Dict], **kwargs) -> str:
        """Chat completion."""
        if not self._connected:
            raise RuntimeError(f"Not connected to {self.NAME}")
        
        # Example API call (customize for your backend):
        # async with self._session.post(
        #     f"{self.url}/v1/chat/completions",
        #     json={"messages": messages, "max_tokens": kwargs.get("max_tokens", 1000)}
        # ) as resp:
        #     data = await resp.json()
        #     return data["choices"][0]["message"]["content"]
        
        raise NotImplementedError("Implement chat() for your backend")


# === Quick Reference: Common Backend APIs ===

# OpenAI-compatible (works with many backends):
#   POST /v1/embeddings
#   POST /v1/completions
#   POST /v1/chat/completions

# Ollama:
#   POST /api/embeddings
#   POST /api/generate
#   POST /api/chat

# vLLM:
#   POST /v1/completions (OpenAI-compatible)
#   POST /v1/chat/completions (OpenAI-compatible)

# HuggingFace Inference API:
#   POST /models/{model_id} (varies by task)

# Add your backend's API reference here for future reference
