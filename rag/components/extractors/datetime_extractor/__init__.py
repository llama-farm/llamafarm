"""DateTime Extractor Component

This component extracts dates, times, and temporal expressions from text using
multiple parsing strategies including dateutil and regex patterns.
"""

from .datetime_extractor import DateTimeExtractor

__all__ = ['DateTimeExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "datetime_extractor"