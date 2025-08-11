"""FaissStore Component

Component for faiss store.
"""

from .faiss_store import FaissStore

__all__ = ['FaissStore']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "store"
COMPONENT_NAME = "faiss_store"
