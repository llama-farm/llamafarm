"""Auto-generated parser registry."""

import json

REGISTRY_DATA = {
    "file_extensions": {
        ".asc": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".ascii": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".bash": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".bat": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".c": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".cfg": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".conf": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".cpp": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".cs": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".csv": [
            {"parser": "CSVParser_Pandas", "priority": 0, "tool": "Pandas"},
            {"parser": "CSVParser_Python", "priority": 0, "tool": "Python csv"},
            {
                "parser": "CSVParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex-Pandas",
            },
        ],
        ".docm": [
            {"parser": "DocxParser_PythonDocx", "priority": 0, "tool": "python-docx"},
            {"parser": "DocxParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".docx": [
            {"parser": "DocxParser_PythonDocx", "priority": 0, "tool": "python-docx"},
            {"parser": "DocxParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".go": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".h": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".ini": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".java": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".js": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".json": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".kt": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".log": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".markdown": [
            {"parser": "MarkdownParser_Python", "priority": 0, "tool": "Python"},
            {
                "parser": "MarkdownParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex",
            },
        ],
        ".md": [
            {"parser": "MarkdownParser_Python", "priority": 0, "tool": "Python"},
            {
                "parser": "MarkdownParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex",
            },
        ],
        ".mdown": [
            {"parser": "MarkdownParser_Python", "priority": 0, "tool": "Python"},
            {
                "parser": "MarkdownParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex",
            },
        ],
        ".mdx": [
            {"parser": "MarkdownParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".mkd": [
            {"parser": "MarkdownParser_Python", "priority": 0, "tool": "Python"},
            {
                "parser": "MarkdownParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex",
            },
        ],
        ".msg": [
            {"parser": "MsgParser_ExtractMsg", "priority": 0, "tool": "extract-msg"}
        ],
        ".pdf": [
            {"parser": "PDFParser_PyPDF2", "priority": 0, "tool": "PyPDF2"},
            {"parser": "PDFParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".php": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".ps1": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".py": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".r": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".rb": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".rs": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".rst": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".rtf": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".scala": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".sh": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".sql": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".swift": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
        ".tex": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".text": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".tsv": [
            {"parser": "CSVParser_Pandas", "priority": 0, "tool": "Pandas"},
            {"parser": "CSVParser_Python", "priority": 0, "tool": "Python csv"},
            {
                "parser": "CSVParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex-Pandas",
            },
        ],
        ".txt": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".xls": [
            {"parser": "ExcelParser_Pandas", "priority": 0, "tool": "Pandas"},
            {
                "parser": "ExcelParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex-Pandas",
            },
        ],
        ".xlsm": [
            {"parser": "ExcelParser_OpenPyXL", "priority": 0, "tool": "OpenPyXL"},
            {"parser": "ExcelParser_Pandas", "priority": 0, "tool": "Pandas"},
            {
                "parser": "ExcelParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex-Pandas",
            },
        ],
        ".xlsx": [
            {"parser": "ExcelParser_OpenPyXL", "priority": 0, "tool": "OpenPyXL"},
            {"parser": "ExcelParser_Pandas", "priority": 0, "tool": "Pandas"},
            {
                "parser": "ExcelParser_LlamaIndex",
                "priority": 0,
                "tool": "LlamaIndex-Pandas",
            },
        ],
        ".xltm": [
            {"parser": "ExcelParser_OpenPyXL", "priority": 0, "tool": "OpenPyXL"}
        ],
        ".xltx": [
            {"parser": "ExcelParser_OpenPyXL", "priority": 0, "tool": "OpenPyXL"}
        ],
        ".xml": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".yaml": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".yml": [
            {"parser": "TextParser_Python", "priority": 0, "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"},
        ],
        ".zsh": [
            {"parser": "TextParser_LlamaIndex", "priority": 0, "tool": "LlamaIndex"}
        ],
    },
    "mime_types": {
        "application/csv": [
            {"parser": "CSVParser_Pandas", "tool": "Pandas"},
            {"parser": "CSVParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "application/json": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "application/octet-stream": [
            {"parser": "MsgParser_ExtractMsg", "tool": "extract-msg"}
        ],
        "application/pdf": [
            {"parser": "PDFParser_PyPDF2", "tool": "PyPDF2"},
            {"parser": "PDFParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "application/vnd.ms-excel": [
            {"parser": "ExcelParser_Pandas", "tool": "Pandas"},
            {"parser": "ExcelParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "application/vnd.ms-excel.sheet.macroEnabled.12": [
            {"parser": "ExcelParser_OpenPyXL", "tool": "OpenPyXL"},
            {"parser": "ExcelParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "application/vnd.ms-outlook": [
            {"parser": "MsgParser_ExtractMsg", "tool": "extract-msg"}
        ],
        "application/vnd.ms-word.document.macroEnabled.12": [
            {"parser": "DocxParser_PythonDocx", "tool": "python-docx"},
            {"parser": "DocxParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
            {"parser": "ExcelParser_OpenPyXL", "tool": "OpenPyXL"},
            {"parser": "ExcelParser_Pandas", "tool": "Pandas"},
            {"parser": "ExcelParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
            {"parser": "DocxParser_PythonDocx", "tool": "python-docx"},
            {"parser": "DocxParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "application/x-pdf": [
            {"parser": "PDFParser_PyPDF2", "tool": "PyPDF2"},
            {"parser": "PDFParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "application/x-yaml": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/csv": [
            {"parser": "CSVParser_Pandas", "tool": "Pandas"},
            {"parser": "CSVParser_Python", "tool": "Python csv"},
            {"parser": "CSVParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "text/javascript": [{"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/markdown": [
            {"parser": "MarkdownParser_Python", "tool": "Python"},
            {"parser": "MarkdownParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/mdx": [{"parser": "MarkdownParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/plain": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/rtf": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/tab-separated-values": [
            {"parser": "CSVParser_Pandas", "tool": "Pandas"},
            {"parser": "CSVParser_Python", "tool": "Python csv"},
            {"parser": "CSVParser_LlamaIndex", "tool": "LlamaIndex-Pandas"},
        ],
        "text/x-c": [{"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/x-chdr": [{"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/x-csrc": [{"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/x-java-source": [
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}
        ],
        "text/x-log": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/x-markdown": [
            {"parser": "MarkdownParser_Python", "tool": "Python"},
            {"parser": "MarkdownParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/x-python": [{"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"}],
        "text/x-rst": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/x-tex": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
        "text/xml": [
            {"parser": "TextParser_Python", "tool": "Python"},
            {"parser": "TextParser_LlamaIndex", "tool": "LlamaIndex"},
        ],
    },
    "parsers": {
        "CSVParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "semantic_chunking",
                "metadata_extraction",
                "field_mapping",
                "row_based_processing",
                "column_based_processing",
                "statistics",
                "data_transformation",
            ],
            "default_config": {
                "chunk_size": 1000,
                "chunk_strategy": "rows",
                "combine_content": True,
                "content_fields": None,
                "content_separator": "\n\n",
                "delimiter": ",",
                "encoding": "utf-8",
                "id_field": None,
                "metadata_fields": [],
                "na_values": ["", "NA", "N/A", "null", "None"],
            },
            "dependencies": {
                "optional": ["numpy"],
                "required": ["llama-index", "llama-index-readers-file", "pandas"],
            },
            "description": "CSV parser using LlamaIndex with Pandas backend for advanced "
            "processing",
            "display_name": "CSV Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/csv",
            "mime_types": ["text/csv", "text/tab-separated-values", "application/csv"],
            "name": "CSVParser_LlamaIndex",
            "parser_type": "csv",
            "supported_extensions": [".csv", ".tsv"],
            "tool": "LlamaIndex-Pandas",
            "version": "1.0.0",
        },
        "CSVParser_Pandas": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "statistics",
                "data_analysis",
                "column_processing",
            ],
            "default_config": {
                "chunk_size": 1000,
                "chunk_strategy": "rows",
                "delimiter": ",",
                "encoding": "utf-8",
                "extract_metadata": True,
                "na_values": ["", "NA", "N/A", "null", "None"],
            },
            "dependencies": {"optional": ["numpy"], "required": ["pandas"]},
            "description": "Advanced CSV parser using Pandas with data analysis capabilities",
            "display_name": "CSV Parser (Pandas)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/csv",
            "mime_types": ["text/csv", "text/tab-separated-values", "application/csv"],
            "name": "CSVParser_Pandas",
            "parser_type": "csv",
            "supported_extensions": [".csv", ".tsv"],
            "tool": "Pandas",
            "version": "1.0.0",
        },
        "CSVParser_Python": {
            "capabilities": ["text_extraction", "chunking", "basic_metadata"],
            "default_config": {
                "chunk_size": 1000,
                "delimiter": ",",
                "encoding": "utf-8",
                "quotechar": '"',
            },
            "dependencies": {"optional": [], "required": []},
            "description": "Simple CSV parser using native Python csv module",
            "display_name": "CSV Parser (Python)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/csv",
            "mime_types": ["text/csv", "text/tab-separated-values"],
            "name": "CSVParser_Python",
            "parser_type": "csv",
            "supported_extensions": [".csv", ".tsv"],
            "tool": "Python csv",
            "version": "1.0.0",
        },
        "DocxParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "semantic_chunking",
                "paragraph_aware_splitting",
                "metadata_extraction",
                "table_extraction",
                "style_preservation",
                "section_detection",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "paragraphs",
                "extract_comments": False,
                "extract_headers_footers": False,
                "extract_images": False,
                "extract_metadata": True,
                "extract_tables": True,
                "preserve_formatting": False,
            },
            "dependencies": {
                "optional": ["python-docx"],
                "required": ["llama-index", "llama-index-readers-file"],
            },
            "description": "Advanced DOCX parser using LlamaIndex with enhanced chunking",
            "display_name": "DOCX Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/docx",
            "mime_types": [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.ms-word.document.macroEnabled.12",
            ],
            "name": "DocxParser_LlamaIndex",
            "parser_type": "docx",
            "supported_extensions": [".docx", ".docm"],
            "tool": "LlamaIndex",
            "version": "1.0.0",
        },
        "DocxParser_PythonDocx": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "table_extraction",
                "header_extraction",
                "footer_extraction",
                "style_preservation",
            ],
            "default_config": {
                "chunk_size": 1000,
                "chunk_strategy": "paragraphs",
                "extract_comments": False,
                "extract_footers": False,
                "extract_headers": True,
                "extract_metadata": True,
                "extract_tables": True,
            },
            "dependencies": {"optional": [], "required": ["python-docx"]},
            "description": "Word document parser using python-docx library",
            "display_name": "DOCX Parser (python-docx)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/docx",
            "mime_types": [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.ms-word.document.macroEnabled.12",
            ],
            "name": "DocxParser_PythonDocx",
            "parser_type": "docx",
            "supported_extensions": [".docx", ".docm"],
            "tool": "python-docx",
            "version": "1.0.0",
        },
        "ExcelParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "semantic_chunking",
                "metadata_extraction",
                "statistics",
                "multi_sheet_processing",
                "sheet_combination",
                "formula_extraction",
                "data_transformation",
            ],
            "default_config": {
                "chunk_size": 1000,
                "chunk_strategy": "rows",
                "combine_sheets": False,
                "extract_formulas": False,
                "extract_metadata": True,
                "header_row": 0,
                "na_values": ["", "NA", "N/A", "null", "None"],
                "sheets": None,
                "skiprows": None,
            },
            "dependencies": {
                "optional": ["xlrd", "tabulate"],
                "required": [
                    "llama-index",
                    "llama-index-readers-file",
                    "pandas",
                    "openpyxl",
                ],
            },
            "description": "Excel parser using LlamaIndex with Pandas backend for "
            "advanced processing",
            "display_name": "Excel Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/excel",
            "mime_types": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
                "application/vnd.ms-excel.sheet.macroEnabled.12",
            ],
            "name": "ExcelParser_LlamaIndex",
            "parser_type": "excel",
            "supported_extensions": [".xlsx", ".xls", ".xlsm"],
            "tool": "LlamaIndex-Pandas",
            "version": "1.0.0",
        },
        "ExcelParser_OpenPyXL": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "formula_extraction",
                "multi_sheet",
                "merged_cells",
            ],
            "default_config": {
                "chunk_size": 1000,
                "data_only": True,
                "extract_formulas": False,
                "extract_metadata": True,
                "sheets": None,
            },
            "dependencies": {"optional": [], "required": ["openpyxl"]},
            "description": "Excel parser using OpenPyXL for XLSX files with formula "
            "support",
            "display_name": "Excel Parser (OpenPyXL)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/excel",
            "mime_types": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel.sheet.macroEnabled.12",
            ],
            "name": "ExcelParser_OpenPyXL",
            "parser_type": "excel",
            "supported_extensions": [".xlsx", ".xlsm", ".xltx", ".xltm"],
            "tool": "OpenPyXL",
            "version": "1.0.0",
        },
        "ExcelParser_Pandas": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "statistics",
                "data_analysis",
                "multi_sheet",
            ],
            "default_config": {
                "chunk_size": 1000,
                "extract_metadata": True,
                "na_values": ["", "NA", "N/A", "null", "None"],
                "sheets": None,
                "skiprows": None,
            },
            "dependencies": {"optional": ["xlrd"], "required": ["pandas", "openpyxl"]},
            "description": "Excel parser using Pandas with data analysis capabilities",
            "display_name": "Excel Parser (Pandas)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/excel",
            "mime_types": [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ],
            "name": "ExcelParser_Pandas",
            "parser_type": "excel",
            "supported_extensions": [".xlsx", ".xls", ".xlsm"],
            "tool": "Pandas",
            "version": "1.0.0",
        },
        "MarkdownParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "semantic_chunking",
                "heading_based_splitting",
                "metadata_extraction",
                "header_hierarchy",
                "code_block_extraction",
                "link_extraction",
                "frontmatter_parsing",
                "preserve_structure",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "headings",
                "extract_code_blocks": True,
                "extract_headings": True,
                "extract_links": True,
                "extract_metadata": True,
                "preserve_formatting": False,
            },
            "dependencies": {
                "optional": ["markdown", "markdown2"],
                "required": ["llama-index", "llama-index-readers-file"],
            },
            "description": "Advanced markdown parser using LlamaIndex with semantic "
            "chunking",
            "display_name": "Markdown Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/markdown",
            "mime_types": ["text/markdown", "text/x-markdown", "text/mdx"],
            "name": "MarkdownParser_LlamaIndex",
            "parser_type": "markdown",
            "supported_extensions": [".md", ".markdown", ".mdown", ".mkd", ".mdx"],
            "tool": "LlamaIndex",
            "version": "1.0.0",
        },
        "MarkdownParser_Python": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "header_extraction",
                "code_block_extraction",
                "link_extraction",
                "frontmatter_parsing",
            ],
            "default_config": {
                "chunk_size": 1000,
                "chunk_strategy": "sections",
                "extract_code_blocks": True,
                "extract_links": True,
                "extract_metadata": True,
            },
            "dependencies": {"optional": [], "required": []},
            "description": "Markdown parser using native Python with regex parsing",
            "display_name": "Markdown Parser (Python)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/markdown",
            "mime_types": ["text/markdown", "text/x-markdown"],
            "name": "MarkdownParser_Python",
            "parser_type": "markdown",
            "supported_extensions": [".md", ".markdown", ".mdown", ".mkd"],
            "tool": "Python",
            "version": "1.0.0",
        },
        "MsgParser_ExtractMsg": {
            "capabilities": [
                "text_extraction",
                "metadata_extraction",
                "attachment_extraction",
                "chunking",
                "email_parsing",
                "header_extraction",
                "body_extraction",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "email_sections",
                "clean_text": True,
                "encoding": "utf-8",
                "extract_attachments": True,
                "extract_headers": True,
                "extract_metadata": True,
                "include_attachment_content": False,
                "preserve_formatting": False,
            },
            "dependencies": {
                "optional": ["chardet", "python-magic"],
                "required": ["extract-msg"],
            },
            "description": "Microsoft Outlook MSG file parser using extract-msg library",
            "display_name": "MSG Parser (extract-msg)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/msg",
            "mime_types": ["application/vnd.ms-outlook", "application/octet-stream"],
            "name": "MsgParser_ExtractMsg",
            "parser_type": "msg",
            "supported_extensions": [".msg"],
            "tool": "extract-msg",
            "version": "1.0.0",
        },
        "PDFParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "semantic_chunking",
                "metadata_extraction",
                "page_aware_splitting",
                "table_extraction",
                "image_extraction",
                "fallback_strategies",
                "ocr_support",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "characters",
                "extract_images": False,
                "extract_metadata": True,
                "extract_tables": False,
                "fallback_strategies": [
                    "llama_pdf_reader",
                    "llama_pymupdf_reader",
                    "direct_pymupdf",
                    "pypdf2_fallback",
                ],
                "preserve_layout": False,
            },
            "dependencies": {
                "optional": ["PyMuPDF", "fitz", "PyPDF2", "pdfplumber", "camelot-py"],
                "required": ["llama-index", "llama-index-readers-file"],
            },
            "description": "Advanced PDF parser using LlamaIndex with multiple fallback "
            "strategies",
            "display_name": "PDF Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/pdf",
            "mime_types": ["application/pdf", "application/x-pdf"],
            "name": "PDFParser_LlamaIndex",
            "parser_type": "pdf",
            "supported_extensions": [".pdf"],
            "tool": "LlamaIndex",
            "version": "1.0.0",
        },
        "PDFParser_PyPDF2": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "layout_preservation",
                "annotation_extraction",
                "page_info_extraction",
                "encryption_handling",
                "form_field_extraction",
                "outline_extraction",
                "image_extraction",
                "xmp_metadata_extraction",
                "rotation_handling",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "paragraphs",
                "clean_text": True,
                "extract_annotations": False,
                "extract_form_fields": False,
                "extract_images": False,
                "extract_links": False,
                "extract_metadata": True,
                "extract_outlines": False,
                "extract_page_info": True,
                "extract_xmp_metadata": False,
                "preserve_layout": True,
            },
            "dependencies": {"optional": ["PyPDF2"], "required": []},
            "description": "Enhanced PDF parser using PyPDF2 with layout preservation and "
            "metadata extraction",
            "display_name": "PDF Parser (PyPDF2)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/pdf",
            "mime_types": ["application/pdf", "application/x-pdf"],
            "name": "PDFParser_PyPDF2",
            "parser_type": "pdf",
            "supported_extensions": [".pdf"],
            "tool": "PyPDF2",
            "version": "1.1.0",
        },
        "TextParser_LlamaIndex": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "semantic_chunking",
                "token_based_chunking",
                "code_aware_parsing",
                "language_detection",
                "structure_preservation",
                "metadata_extraction",
                "json_parsing",
                "yaml_parsing",
                "xml_parsing",
                "code_syntax_parsing",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "semantic",
                "clean_text": True,
                "detect_language": True,
                "encoding": "utf-8",
                "extract_metadata": True,
                "include_prev_next_rel": True,
                "preserve_code_structure": True,
                "semantic_breakpoint_percentile_threshold": 95,
                "semantic_buffer_size": 1,
                "token_model": "gpt-3.5-turbo",
            },
            "dependencies": {
                "optional": ["transformers", "tiktoken", "tree-sitter", "pygments"],
                "required": ["llama-index", "llama-index-core"],
            },
            "description": "Advanced text parser using LlamaIndex with semantic "
            "splitting and multi-format support",
            "display_name": "Text Parser (LlamaIndex)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/text",
            "mime_types": [
                "text/plain",
                "text/x-log",
                "text/x-rst",
                "text/x-tex",
                "text/rtf",
                "application/x-yaml",
                "application/json",
                "text/xml",
                "text/x-python",
                "text/javascript",
                "text/x-java-source",
                "text/x-c",
                "text/x-csrc",
                "text/x-chdr",
            ],
            "name": "TextParser_LlamaIndex",
            "parser_type": "text",
            "supported_extensions": [
                ".txt",
                ".text",
                ".log",
                ".rst",
                ".tex",
                ".rtf",
                ".asc",
                ".ascii",
                ".conf",
                ".cfg",
                ".ini",
                ".yaml",
                ".yml",
                ".json",
                ".xml",
                ".py",
                ".js",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".php",
                ".rb",
                ".go",
                ".rs",
                ".swift",
                ".kt",
                ".scala",
                ".r",
                ".sql",
                ".sh",
                ".bash",
                ".zsh",
                ".ps1",
                ".bat",
            ],
            "tool": "LlamaIndex",
            "version": "1.0.0",
        },
        "TextParser_Python": {
            "capabilities": [
                "text_extraction",
                "chunking",
                "encoding_detection",
                "metadata_extraction",
                "basic_statistics",
            ],
            "default_config": {
                "chunk_overlap": 100,
                "chunk_size": 1000,
                "chunk_strategy": "sentences",
                "clean_text": True,
                "encoding": "utf-8",
                "extract_metadata": True,
            },
            "dependencies": {"optional": ["chardet"], "required": []},
            "description": "Text parser using native Python with encoding detection",
            "display_name": "Text Parser (Python)",
            "implementation_dir": "/Users/mhamann/projects/llamafarm/llamafarm/rag/components/parsers/text",
            "mime_types": [
                "text/plain",
                "text/x-log",
                "text/x-rst",
                "text/x-tex",
                "text/rtf",
                "application/x-yaml",
                "application/json",
                "text/xml",
            ],
            "name": "TextParser_Python",
            "parser_type": "text",
            "supported_extensions": [
                ".txt",
                ".text",
                ".log",
                ".rst",
                ".tex",
                ".rtf",
                ".asc",
                ".ascii",
                ".conf",
                ".cfg",
                ".ini",
                ".yaml",
                ".yml",
                ".json",
                ".xml",
            ],
            "tool": "Python",
            "version": "1.0.0",
        },
    },
    "tools": {
        "LlamaIndex": [
            {
                "capabilities": [
                    "text_extraction",
                    "semantic_chunking",
                    "heading_based_splitting",
                    "metadata_extraction",
                    "header_hierarchy",
                    "code_block_extraction",
                    "link_extraction",
                    "frontmatter_parsing",
                    "preserve_structure",
                ],
                "parser": "MarkdownParser_LlamaIndex",
                "type": "markdown",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "semantic_chunking",
                    "metadata_extraction",
                    "page_aware_splitting",
                    "table_extraction",
                    "image_extraction",
                    "fallback_strategies",
                    "ocr_support",
                ],
                "parser": "PDFParser_LlamaIndex",
                "type": "pdf",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "semantic_chunking",
                    "token_based_chunking",
                    "code_aware_parsing",
                    "language_detection",
                    "structure_preservation",
                    "metadata_extraction",
                    "json_parsing",
                    "yaml_parsing",
                    "xml_parsing",
                    "code_syntax_parsing",
                ],
                "parser": "TextParser_LlamaIndex",
                "type": "text",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "semantic_chunking",
                    "paragraph_aware_splitting",
                    "metadata_extraction",
                    "table_extraction",
                    "style_preservation",
                    "section_detection",
                ],
                "parser": "DocxParser_LlamaIndex",
                "type": "docx",
            },
        ],
        "LlamaIndex-Pandas": [
            {
                "capabilities": [
                    "text_extraction",
                    "semantic_chunking",
                    "metadata_extraction",
                    "statistics",
                    "multi_sheet_processing",
                    "sheet_combination",
                    "formula_extraction",
                    "data_transformation",
                ],
                "parser": "ExcelParser_LlamaIndex",
                "type": "excel",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "semantic_chunking",
                    "metadata_extraction",
                    "field_mapping",
                    "row_based_processing",
                    "column_based_processing",
                    "statistics",
                    "data_transformation",
                ],
                "parser": "CSVParser_LlamaIndex",
                "type": "csv",
            },
        ],
        "OpenPyXL": [
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "formula_extraction",
                    "multi_sheet",
                    "merged_cells",
                ],
                "parser": "ExcelParser_OpenPyXL",
                "type": "excel",
            }
        ],
        "Pandas": [
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "statistics",
                    "data_analysis",
                    "multi_sheet",
                ],
                "parser": "ExcelParser_Pandas",
                "type": "excel",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "statistics",
                    "data_analysis",
                    "column_processing",
                ],
                "parser": "CSVParser_Pandas",
                "type": "csv",
            },
        ],
        "PyPDF2": [
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "layout_preservation",
                    "annotation_extraction",
                    "page_info_extraction",
                    "encryption_handling",
                    "form_field_extraction",
                    "outline_extraction",
                    "image_extraction",
                    "xmp_metadata_extraction",
                    "rotation_handling",
                ],
                "parser": "PDFParser_PyPDF2",
                "type": "pdf",
            }
        ],
        "Python": [
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "header_extraction",
                    "code_block_extraction",
                    "link_extraction",
                    "frontmatter_parsing",
                ],
                "parser": "MarkdownParser_Python",
                "type": "markdown",
            },
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "encoding_detection",
                    "metadata_extraction",
                    "basic_statistics",
                ],
                "parser": "TextParser_Python",
                "type": "text",
            },
        ],
        "Python csv": [
            {
                "capabilities": ["text_extraction", "chunking", "basic_metadata"],
                "parser": "CSVParser_Python",
                "type": "csv",
            }
        ],
        "extract-msg": [
            {
                "capabilities": [
                    "text_extraction",
                    "metadata_extraction",
                    "attachment_extraction",
                    "chunking",
                    "email_parsing",
                    "header_extraction",
                    "body_extraction",
                ],
                "parser": "MsgParser_ExtractMsg",
                "type": "msg",
            }
        ],
        "python-docx": [
            {
                "capabilities": [
                    "text_extraction",
                    "chunking",
                    "metadata_extraction",
                    "table_extraction",
                    "header_extraction",
                    "footer_extraction",
                    "style_preservation",
                ],
                "parser": "DocxParser_PythonDocx",
                "type": "docx",
            }
        ],
    },
    "version": "1.0",
}


class ParserRegistry:
    """Parser registry with lookup methods."""

    def __init__(self):
        self.data = REGISTRY_DATA

    def get_parser(self, name: str) -> dict:
        """Get parser configuration by name."""
        return self.data["parsers"].get(name)

    def get_parsers_for_extension(self, ext: str) -> list:
        """Get parsers that support a file extension."""
        return self.data["file_extensions"].get(ext, [])

    def get_parsers_for_mime(self, mime: str) -> list:
        """Get parsers that support a MIME type."""
        return self.data["mime_types"].get(mime, [])

    def get_parsers_by_tool(self, tool: str) -> list:
        """Get all parsers using a specific tool."""
        return self.data["tools"].get(tool, [])

    def list_all_parsers(self) -> list:
        """List all available parser names."""
        return list(self.data["parsers"].keys())


registry = ParserRegistry()
