"""Link Extractor Component

This component extracts URLs, email addresses, phone numbers, and other references.
"""

from .link_extractor import LinkExtractor

__all__ = ['LinkExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "link_extractor"