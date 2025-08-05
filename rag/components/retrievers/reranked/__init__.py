"""Reranked Component

Component for reranked.
"""

from .reranked import Reranked

__all__ = ['Reranked']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "reranked"
