"""Entity Extractor Component

This component extracts named entities from text using NLP models with regex fallback.
"""

from .entity_extractor import EntityExtractor

__all__ = ['EntityExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "entity_extractor"