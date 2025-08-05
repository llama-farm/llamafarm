"""PineconeStore Component

Component for pinecone store.
"""

from .pinecone_store import PineconeStore

__all__ = ['PineconeStore']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "store"
COMPONENT_NAME = "pinecone_store"
