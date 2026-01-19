"""Shared RAG components library for LlamaFarm.

This library provides reusable RAG components that are used by both:
- server/ (lightweight RAG with embedded ChromaDB)
- rag/ (advanced RAG with multiple vector stores)

Components:
- embedders: HTTP-based embedders (Universal, Ollama)
- retrievers: Retrieval strategies (BasicSimilarity, etc.)
- stores: Vector stores (ChromaStore)
- models: Shared data models (Document, RetrievalResult, etc.)
"""

__version__ = "0.1.0"

# Import order matters - models first, then implementations
from .models import Document, EmbeddingVector, RetrievalResult

# Embedders
from .embedders.base import Embedder
from .embedders.ollama_embedder import OllamaEmbedder
from .embedders.universal_embedder import UniversalEmbedder

# Retrievers
from .retrievers.base import RetrievalStrategy
from .retrievers.basic_similarity import BasicSimilarityStrategy

# Stores
from .stores.base import VectorStore
from .stores.chroma_store import ChromaStore

__all__ = [
    # Models
    "Document",
    "EmbeddingVector",
    "RetrievalResult",
    # Embedders
    "Embedder",
    "UniversalEmbedder",
    "OllamaEmbedder",
    # Retrievers
    "RetrievalStrategy",
    "BasicSimilarityStrategy",
    # Stores
    "VectorStore",
    "ChromaStore",
]
