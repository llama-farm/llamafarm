"""
Schema-driven extractors with self-contained components.

This module provides various extractors organized in a schema-driven architecture
where each extractor is a self-contained component with its own configuration.
"""

from .base import BaseExtractor, ExtractorRegistry, ExtractorPipeline

# Create a global registry instance
registry = ExtractorRegistry()

# Individual extractors can be imported as needed from their subdirectories
# Example: from components.extractors.entity_extractor.entity_extractor import EntityExtractor

__all__ = ["BaseExtractor", "ExtractorRegistry", "ExtractorPipeline", "registry"]