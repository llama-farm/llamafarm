"""MarkdownParser Component

Component for markdown parser.
"""

from .markdown_parser import MarkdownParser

__all__ = ['MarkdownParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "markdown_parser"
