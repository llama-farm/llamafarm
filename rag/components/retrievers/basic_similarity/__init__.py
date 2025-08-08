"""BasicSimilarity Component

Component for basic similarity.
"""

from .basic_similarity import BasicSimilarityStrategy

__all__ = ['BasicSimilarityStrategy']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "basic_similarity"
