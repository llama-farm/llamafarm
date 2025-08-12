"""Reranked Component

Component for reranked.
"""

from .reranked import RerankedStrategy

__all__ = ['RerankedStrategy']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "reranked"
