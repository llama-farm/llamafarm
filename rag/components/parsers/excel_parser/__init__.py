"""ExcelParser Component

Component for excel parser.
"""

from .excel_parser import ExcelParser

__all__ = ['ExcelParser']

# Component metadata (read from schema.json at runtime)
COMPONENT_TYPE = "parser"
COMPONENT_NAME = "excel_parser"
