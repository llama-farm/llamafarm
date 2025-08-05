"""Plain Text Parser Component

This component parses plain text files with encoding detection and structure analysis.
"""

from .text_parser import PlainTextParser

__all__ = ['PlainTextParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "text_parser"