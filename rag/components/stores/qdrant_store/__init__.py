"""QdrantStore Component

Component for qdrant store.
"""

from .qdrant_store import QdrantStore

__all__ = ['QdrantStore']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "store"
COMPONENT_NAME = "qdrant_store"
