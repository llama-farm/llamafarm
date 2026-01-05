"""Entity Extractor Component

This component extracts named entities from text using NLP models with regex fallback.

Phase 18: Enhanced with:
- Entity and Relationship dataclasses for structured output
- Graph store integration via extract_entities_to_graph()
- Optional LLM-based relationship extraction
"""

from .entity_extractor import (
    Entity,
    EntityExtractor,
    Relationship,
    extract_entities_to_graph,
)

__all__ = [
    "Entity",
    "EntityExtractor",
    "Relationship",
    "extract_entities_to_graph",
]

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "entity_extractor"
