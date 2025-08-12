"""OllamaEmbedder Component

Component for ollama embedder.
"""

from .ollama_embedder import OllamaEmbedder

__all__ = ['OllamaEmbedder']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "embedder"
COMPONENT_NAME = "ollama_embedder"
