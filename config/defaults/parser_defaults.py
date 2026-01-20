"""Built-in default configurations for parser types."""

from __future__ import annotations

from typing import Any

PARSER_DEFAULTS: dict[str, dict[str, Any]] = {
    "PDFParser_LlamaIndex": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "extract_images": False,
    },
    "PDFParser_PyPDF2": {
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "MarkdownParser_Python": {
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "TextParser_Python": {
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "CSVParser_Pandas": {
        "delimiter": ",",
        "has_header": True,
    },
    "ExcelParser_OpenPyXL": {
        "sheet_names": None,
    },
    "DocxParser_PythonDocx": {
        "include_tables": True,
    },
}


def get_parser_defaults(parser_type: str) -> dict[str, Any]:
    """Return default configuration for a parser type."""
    return PARSER_DEFAULTS.get(parser_type, {}).copy()
