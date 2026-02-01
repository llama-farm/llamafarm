"""
Embedding engine for semantic routing.

Generates 768-dimensional embeddings using nomic-embed-text model.
Supports Ollama (local) and LlamaFarm (cloud) backends.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import logging

import numpy as np
import aiohttp

logger = logging.getLogger(__name__)


class EmbeddingBackend(Enum):
    """Available embedding backends."""
    OLLAMA = "ollama"
    LLAMAFARM = "llamafarm"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding engine."""
    backend: EmbeddingBackend = EmbeddingBackend.OLLAMA
    model: str = "nomic-embed-text"
    dimension: int = 768
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    llamafarm_url: str = "https://llamafarm.dev/api"
    llamafarm_api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaProvider(EmbeddingProvider):
    """Ollama local embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.base_url = f"http://{config.ollama_host}:{config.ollama_port}"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if Ollama is running and has the model."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                # Check for nomic-embed-text or nomic-embed-text:latest
                has_model = any(
                    self.config.model in m or m.startswith(self.config.model)
                    for m in models
                )
                if not has_model:
                    logger.warning(
                        f"Model {self.config.model} not found in Ollama. "
                        f"Available: {models}"
                    )
                return has_model
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama."""
        session = await self._get_session()

        payload = {
            "model": self.config.model,
            "prompt": text
        }

        for attempt in range(self.config.max_retries):
            try:
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"Ollama error: {error}")

                    data = await resp.json()
                    embedding = np.array(data["embedding"], dtype=np.float32)

                    if len(embedding) != self.config.dimension:
                        logger.warning(
                            f"Unexpected embedding dimension: {len(embedding)}, "
                            f"expected {self.config.dimension}"
                        )

                    return embedding

            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Ollama request failed: {e}")

        raise RuntimeError("Max retries exceeded")

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't have native batch, so we parallelize
        tasks = [self.embed(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return np.stack(results)


class LlamaFarmProvider(EmbeddingProvider):
    """LlamaFarm cloud embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.api_key = config.llamafarm_api_key or os.environ.get("LLAMAFARM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LlamaFarm API key required. Set LLAMAFARM_API_KEY env var "
                "or pass llamafarm_api_key in config."
            )
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if LlamaFarm is accessible."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.llamafarm_url}/health") as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"LlamaFarm health check failed: {e}")
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding using LlamaFarm."""
        session = await self._get_session()

        payload = {
            "model": "nomic-embed-text",
            "input": text
        }

        for attempt in range(self.config.max_retries):
            try:
                async with session.post(
                    f"{self.config.llamafarm_url}/embed",
                    json=payload
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        raise RuntimeError(f"LlamaFarm error: {error}")

                    data = await resp.json()
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    return embedding

            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"LlamaFarm request failed: {e}")

        raise RuntimeError("Max retries exceeded")

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        session = await self._get_session()

        payload = {
            "model": "nomic-embed-text",
            "input": texts
        }

        async with session.post(
            f"{self.config.llamafarm_url}/embed",
            json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"LlamaFarm error: {error}")

            data = await resp.json()
            embeddings = np.array(data["embeddings"], dtype=np.float32)
            return embeddings


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
        self._provider: Optional[EmbeddingProvider] = None
        self._initialized = False
        self._cache: dict[str, np.ndarray] = {}
        self._cache_max_size = 1000

    async def initialize(self) -> None:
        """Initialize the embedding engine, selecting best available backend."""
        if self._initialized:
            return

        # Try Ollama first (local, no API key needed)
        ollama = OllamaProvider(self.config)
        if await ollama.health_check():
            logger.info("Using Ollama backend for embeddings")
            self._provider = ollama
            self._initialized = True
            return

        # Fallback to LlamaFarm
        try:
            llamafarm = LlamaFarmProvider(self.config)
            if await llamafarm.health_check():
                logger.info("Using LlamaFarm backend for embeddings")
                self._provider = llamafarm
                self._initialized = True
                return
        except ValueError:
            pass  # No API key

        raise RuntimeError(
            "No embedding backend available. Either:\n"
            "1. Install and run Ollama with: ollama pull nomic-embed-text\n"
            "2. Set LLAMAFARM_API_KEY environment variable"
        )

    async def close(self) -> None:
        """Close the embedding engine."""
        if self._provider:
            await self._provider.close()
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
            normalize: Whether to L2 normalize the result (default: True)

        Returns:
            768-dimensional embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Check cache
        cache_key = text[:200]  # Truncate for cache key
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        vec = await self._provider.embed(text)

        if normalize:
            vec = self._normalize(vec)

        # Update cache with LRU-like behavior
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
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

        Args:
            texts: List of input texts
            normalize: Whether to L2 normalize results

        Returns:
            Array of shape (N, 768) with embeddings
        """
        if not self._initialized:
            await self.initialize()

        vecs = await self._provider.embed_batch(texts)

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vecs = vecs / norms

        return vecs

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Assumes vectors are already normalized. If not, use cosine_similarity_unnormalized.

        Returns:
            Similarity score in range [-1, 1]
        """
        return float(np.dot(vec1, vec2))

    @staticmethod
    def cosine_similarity_unnormalized(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity for unnormalized vectors.

        Returns:
            Similarity score in range [-1, 1]
        """
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
        """
        Compute cosine similarity between query and multiple candidates.

        Args:
            query: Single vector of shape (768,)
            candidates: Array of shape (N, 768)

        Returns:
            Array of similarities, shape (N,)
        """
        # Assumes normalized vectors
        return candidates @ query

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.config.dimension

    @property
    def backend(self) -> Optional[str]:
        """Return current backend name."""
        if isinstance(self._provider, OllamaProvider):
            return "ollama"
        elif isinstance(self._provider, LlamaFarmProvider):
            return "llamafarm"
        return None
