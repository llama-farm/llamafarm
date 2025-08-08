"""HybridUniversal Component

Component for hybrid universal.
"""

from .hybrid_universal import HybridUniversalStrategy

__all__ = ['HybridUniversalStrategy']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "retriever"
COMPONENT_NAME = "hybrid_universal"
