"""SentenceTransformerEmbedder Component

Component for sentence transformer embedder.
"""

from .sentence_transformer_embedder import SentenceTransformerEmbedder

__all__ = ['SentenceTransformerEmbedder']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "embedder"
COMPONENT_NAME = "sentence_transformer_embedder"
