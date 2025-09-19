"""
Schema-driven retrieval strategies with self-contained components.

This module provides various retrieval strategies organized in a schema-driven architecture
where each strategy is a self-contained component with its own configuration.
"""

from .base import RetrievalStrategy, RetrievalResult, HybridRetrievalStrategy

# Individual strategies can be imported as needed from their subdirectories
# Example: from rag.components.retrievers.basic_similarity.basic_similarity import BasicSimilarityStrategy

__all__ = ["RetrievalStrategy", "RetrievalResult", "HybridRetrievalStrategy"]