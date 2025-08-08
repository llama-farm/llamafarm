"""Table Extractor Component

This component extracts structured tabular data from documents using multiple format detection methods.
"""

from .table_extractor import TableExtractor

__all__ = ['TableExtractor']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "extractor"
COMPONENT_NAME = "table_extractor"