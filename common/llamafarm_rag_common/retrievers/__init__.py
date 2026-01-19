"""Retrieval strategies for shared RAG library."""

from .base import RetrievalStrategy
from .basic_similarity import BasicSimilarityStrategy

__all__ = ["RetrievalStrategy", "BasicSimilarityStrategy"]
