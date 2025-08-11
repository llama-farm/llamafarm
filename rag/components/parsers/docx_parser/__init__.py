"""DocxParser Component

Component for docx parser.
"""

from .docx_parser import DocxParser

__all__ = ['DocxParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "docx_parser"
