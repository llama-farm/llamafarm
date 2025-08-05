"""Pattern Extractor Component

This component extracts regex patterns from text (emails, phones, URLs, etc.).
"""

from .pattern_extractor import PatternExtractor

__all__ = ['PatternExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "pattern_extractor"