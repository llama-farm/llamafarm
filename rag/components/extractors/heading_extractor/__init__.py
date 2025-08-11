"""Heading Extractor Component

This component extracts document headings and hierarchical structure.
"""

from .heading_extractor import HeadingExtractor

__all__ = ['HeadingExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "heading_extractor"