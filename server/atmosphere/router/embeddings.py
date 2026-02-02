"""
Embedding engine for semantic routing.

Generates embeddings using available backends (Ollama, LlamaFarm, etc.)
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..discovery.ollama import OllamaBackend, OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding engine."""
    model: str = "nomic-embed-text"
    dimension: int = 768
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    timeout: float = 30.0
    cache_size: int = 1000


class EmbeddingEngine:
    """
    Main embedding engine with automatic backend selection.
    
    Usage:
        engine = EmbeddingEngine()
        await engine.initialize()
        
        vec = await engine.embed("analyze this image for objects")
        similarity = engine.cosine_similarity(vec1, vec2)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._backend: Optional[OllamaBackend] = None
        self._initialized = False
        self._cache: Dict[str, np.ndarray] = {}

    async def initialize(self) -> None:
        """Initialize the embedding engine."""
        if self._initialized:
            return

        # Try Ollama
        ollama_config = OllamaConfig(
            host=self.config.ollama_host,
            port=self.config.ollama_port,
            timeout=self.config.timeout,
            embedding_model=self.config.model
        )
        
        ollama = OllamaBackend(ollama_config)
        
        if await ollama.health_check():
            # Verify embedding model is available
            if await ollama.has_model(self.config.model):
                self._backend = ollama
                self._initialized = True
                logger.info(f"Using Ollama backend with {self.config.model}")
                return
            else:
                logger.warning(f"Model {self.config.model} not found in Ollama")

        raise RuntimeError(
            "No embedding backend available. Run:\n"
            "  ollama pull nomic-embed-text"
        )

    async def close(self) -> None:
        """Close the embedding engine."""
        if self._backend:
            await self._backend.close()
        self._initialized = False

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    async def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text to embed
            normalize: Whether to L2 normalize the result
            
        Returns:
            Embedding vector as numpy array
        """
        if not self._initialized:
            await self.initialize()

        # Check cache
        cache_key = text[:200]
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        embedding = await self._backend.embed(text)
        vec = np.array(embedding, dtype=np.float32)

        if normalize:
            vec = self._normalize(vec)

        # Update cache
        if len(self._cache) >= self.config.cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[cache_key] = vec.copy()

        return vec

    async def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Returns:
            Array of shape (N, dimension) with embeddings
        """
        if not self._initialized:
            await self.initialize()

        embeddings = await self._backend.embed_batch(texts)
        vecs = np.array(embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms

        return vecs

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Assumes vectors are already normalized.
        """
        return float(np.dot(vec1, vec2))

    @staticmethod
    def cosine_similarity_unnormalized(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity for unnormalized vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    @staticmethod
    def batch_cosine_similarity(
        query: np.ndarray,
        candidates: np.ndarray
    ) -> np.ndarray:
        """Compute similarity between query and multiple candidates."""
        return candidates @ query

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.config.dimension

    @property
    def backend(self) -> Optional[str]:
        """Return current backend name."""
        if self._backend:
            return "ollama"
        return None
