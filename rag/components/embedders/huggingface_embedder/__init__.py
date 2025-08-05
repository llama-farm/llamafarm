"""HuggingfaceEmbedder Component

Component for huggingface embedder.
"""

from .huggingface_embedder import HuggingfaceEmbedder

__all__ = ['HuggingfaceEmbedder']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "embedder"
COMPONENT_NAME = "huggingface_embedder"
