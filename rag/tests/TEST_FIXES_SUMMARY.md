# Parser Test Fixes Summary

## Overview
Updated the parser tests to be compatible with the refactored parser system that uses tool-specific parsers.

## Key Changes Made

### 1. Parser Naming Convention
The new parser system uses tool-specific naming:
- `PDFParser_PyPDF2` instead of generic `PDFParser`
- `CSVParser_Pandas` and `CSVParser_Python` instead of single `CSVParser`
- `TextParser_Python` and `TextParser_LlamaIndex` for text parsing
- `ExcelParser_OpenPyXL` and `ExcelParser_Pandas` for Excel files

### 2. Metadata Fields
Parsers now set metadata fields as:
- `"parser"` field contains the parser name (e.g., "CSVParser_Pandas")
- `"tool"` field contains the tool used (e.g., "Pandas", "PyPDF2")
- Previous tests were looking for `"parser_type"` which doesn't exist

### 3. Factory Methods
Updated to use correct factory methods:
- `ParserFactory.create_parser()` for backward compatibility
- `ToolAwareParserFactory.create_parser()` with `parser_name` or `file_type` parameters
- `ToolAwareParserFactory.list_parsers()` to get available parsers
- `ToolAwareParserFactory.discover_parsers()` to discover parser configurations

### 4. Registry Methods
The `ParserRegistry` class provides:
- `get_parsers_for_extension()` - Get parsers for a file extension
- `get_parsers_for_mime()` - Get parsers for a MIME type
- `list_all_parsers()` - List all available parser names

## Test Files

### Original Test Files (Outdated)
- `test_parsers.py` - Uses old imports and class names
- `test_new_parsers.py` - Uses old import paths

### Fixed Test File
- `test_parsers_fixed.py` - Updated with correct imports, class names, and metadata fields

## Test Results
All 13 tests now pass successfully:
- ✅ CSV parsers (Pandas and Python)
- ✅ PDF parser (PyPDF2)
- ✅ DOCX parser (python-docx)
- ✅ Markdown parser (Python)
- ✅ Text parsers (Python and LlamaIndex)
- ✅ Excel parsers (OpenPyXL and Pandas)
- ✅ Parser factory tests
- ✅ Tool-aware factory tests
- ✅ Parser registry tests
- ✅ Chunking functionality tests

## Running the Tests
```bash
cd rag
uv run pytest tests/test_parsers_fixed.py -v
```

## Notes
- Some parsers require additional dependencies (pandas, openpyxl, python-docx, llama-index)
- Tests gracefully skip if dependencies are not installed
- The parser system is now more modular and allows for multiple implementations per file type