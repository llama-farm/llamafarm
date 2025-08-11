"""ChromaStore Component

Component for chroma store.
"""

from .chroma_store import ChromaStore

__all__ = ['ChromaStore']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "store"
COMPONENT_NAME = "chroma_store"
