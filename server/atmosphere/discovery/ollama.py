"""
Ollama backend for embeddings.

Provides embedding capabilities via local Ollama instance.
"""

from dataclasses import dataclass
from typing import List, Optional
import aiohttp
import logging

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""
    host: str = "localhost"
    port: int = 11434
    model: str = "nomic-embed-text"


class OllamaBackend:
    """
    Ollama embedding backend.
    
    Uses Ollama's /api/embeddings endpoint for text embeddings.
    """
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"
    
    async def connect(self) -> bool:
        """Connect to Ollama."""
        try:
            self._session = aiohttp.ClientSession()
            async with self._session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    logger.info(f"Connected to Ollama at {self.base_url}")
                    return True
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        return False
    
    async def disconnect(self):
        """Disconnect from Ollama."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.config.model, "prompt": text}
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Embedding failed: {resp.status}")
            data = await resp.json()
            return data.get("embedding", [])
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts."""
        return [await self.embed(t) for t in texts]
