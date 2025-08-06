"""HtmlParser Component

Component for html parser.
"""

from .html_parser import HtmlParser as HTMLParser

__all__ = ['HTMLParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "html_parser"
