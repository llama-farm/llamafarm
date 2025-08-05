"""PdfParser Component

Component for pdf parser.
"""

from .pdf_parser import PDFParser

__all__ = ['PDFParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "pdf_parser"
