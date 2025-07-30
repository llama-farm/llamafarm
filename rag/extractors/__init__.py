"""
Local-only extractors for metadata enhancement without requiring LLMs.

This module provides various extractors that work without external dependencies
on language models, making them suitable for local processing and privacy-sensitive
environments.
"""

from .base import BaseExtractor, ExtractorRegistry
from .keyword_extractors import RAKEExtractor, YAKEExtractor, TFIDFExtractor
from .entity_extractor import EntityExtractor
from .datetime_extractor import DateTimeExtractor
from .statistics_extractor import ContentStatisticsExtractor

# Register all extractors
registry = ExtractorRegistry()

# Keyword extractors
registry.register("rake", RAKEExtractor)
registry.register("yake", YAKEExtractor)
registry.register("tfidf", TFIDFExtractor)

# Entity extractors
registry.register("entities", EntityExtractor)

# Content extractors
registry.register("datetime", DateTimeExtractor)
registry.register("statistics", ContentStatisticsExtractor)

__all__ = [
    "BaseExtractor",
    "ExtractorRegistry", 
    "RAKEExtractor",
    "YAKEExtractor", 
    "TFIDFExtractor",
    "EntityExtractor",
    "DateTimeExtractor",
    "ContentStatisticsExtractor",
    "registry"
]