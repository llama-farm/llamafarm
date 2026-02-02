"""
Discovery adapters for Atmosphere.

Discovers and integrates with:
- LlamaFarm (local inference, embeddings, RAG)
- Ollama (fallback local LLM)
- Other Atmosphere nodes (via gossip)
"""

from .llamafarm import LlamaFarmAdapter
from .ollama import OllamaBackend, OllamaConfig

__all__ = ["LlamaFarmAdapter", "OllamaBackend", "OllamaConfig"]
