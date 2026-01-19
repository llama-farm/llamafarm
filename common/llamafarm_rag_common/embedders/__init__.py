"""Embedder implementations for shared RAG library."""

from .base import Embedder
from .ollama_embedder import OllamaEmbedder
from .universal_embedder import UniversalEmbedder

__all__ = ["Embedder", "UniversalEmbedder", "OllamaEmbedder"]
