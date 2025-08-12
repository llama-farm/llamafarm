"""CSV Parser Component

This component parses CSV files with configurable field mapping for various use cases.
"""

from .csv_parser import CSVParser, CustomerSupportCSVParser

__all__ = ['CSVParser', 'CustomerSupportCSVParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "csv_parser"