"""Keyword Extractor Component

This component extracts keywords and key phrases using multiple algorithms
including RAKE, YAKE, and TF-IDF with options to combine results.
"""

from .keyword_extractor import KeywordExtractor, RAKEExtractor, YAKEExtractor, TFIDFExtractor

__all__ = ['KeywordExtractor', 'RAKEExtractor', 'YAKEExtractor', 'TFIDFExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "keyword_extractor"