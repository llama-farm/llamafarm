"""OpenaiEmbedder Component

Component for openai embedder.
"""

from .openai_embedder import OpenaiEmbedder

__all__ = ['OpenaiEmbedder']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "embedder"
COMPONENT_NAME = "openai_embedder"
