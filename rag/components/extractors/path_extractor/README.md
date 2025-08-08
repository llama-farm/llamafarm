# Path Extractor Component

A specialized extractor for preserving source file path information in document metadata, ensuring reliable retrieval and display in RAG systems.

## Overview

The Path Extractor solves a common problem in RAG systems: maintaining document source information throughout the pipeline. When documents are processed, embedded, and stored in vector databases, the original file path information can be lost, resulting in "unknown" sources in search results.

## Features

- **Comprehensive Path Storage**: Store full paths, filenames, directories, and extensions
- **Cross-Platform Support**: Normalize path separators for consistent handling
- **Relative Path Support**: Convert absolute paths to relative for portability
- **Flexible Configuration**: Choose which path components to store
- **Dual Storage**: Stores in both extractor metadata and top-level fields for compatibility

## Installation

The Path Extractor is included in the RAG system. No additional dependencies are required.

## Quick Start

```python
from components.extractors.path_extractor import PathExtractor

# Create extractor with default settings
path_extractor = PathExtractor("path_extractor")

# Process documents
documents = parser.parse("path/to/file.pdf")
for doc in documents:
    doc.source = "path/to/file.pdf"  # Ensure source is set

documents = path_extractor.extract(documents)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `store_full_path` | bool | true | Store the complete file path |
| `store_filename` | bool | true | Store just the filename |
| `store_directory` | bool | true | Store the parent directory |
| `store_extension` | bool | true | Store the file extension |
| `normalize_paths` | bool | false | Convert backslashes to forward slashes |
| `relative_to` | string | null | Make paths relative to this directory |

## Usage Examples

### Basic Usage (All Path Components)
```python
extractor = PathExtractor("path_extractor", {
    "store_full_path": True,
    "store_filename": True,
    "store_directory": True,
    "store_extension": True
})
```

### Minimal Storage (Filename Only)
```python
extractor = PathExtractor("path_extractor", {
    "store_full_path": False,
    "store_filename": True,
    "store_directory": False,
    "store_extension": False
})
```

### Cross-Platform Web Application
```python
extractor = PathExtractor("path_extractor", {
    "store_full_path": True,
    "store_filename": True,
    "normalize_paths": True,
    "relative_to": "./uploads"
})
```

## Integration with RAG Pipeline

The Path Extractor should be applied **first** in the extraction pipeline to ensure source information is available for other extractors:

```python
# 1. Parse documents
documents = parser.parse(file_path)

# 2. Set source on each document
for doc in documents:
    doc.source = file_path

# 3. Apply path extractor FIRST
documents = path_extractor.extract(documents)

# 4. Apply other extractors
documents = entity_extractor.extract(documents)
documents = summary_extractor.extract(documents)

# 5. Generate embeddings and store
# The path information will be preserved in the vector database
```

## Output Format

The extractor adds path information to document metadata in two locations:

1. **Extractor Metadata** (`doc.metadata["extractors"]["path"]`):
```json
{
  "full_path": "/path/to/document.pdf",
  "filename": "document.pdf",
  "directory": "/path/to",
  "extension": ".pdf"
}
```

2. **Top-Level Metadata** (for compatibility):
- `doc.metadata["file_path"]` - Full path
- `doc.metadata["file_name"]` - Filename
- `doc.metadata["directory"]` - Directory path
- `doc.metadata["file_extension"]` - File extension

## Common Issues and Solutions

### Issue: Source showing as "unknown" in search results
**Solution**: Ensure you're setting `doc.source` before applying the extractor:
```python
for doc in documents:
    doc.source = file_path  # This is required!
```

### Issue: Paths not portable across systems
**Solution**: Use normalized relative paths:
```python
extractor = PathExtractor("path_extractor", {
    "normalize_paths": True,
    "relative_to": "./"
})
```

### Issue: Windows paths showing backslashes in web UI
**Solution**: Enable path normalization:
```python
extractor = PathExtractor("path_extractor", {
    "normalize_paths": True
})
```

## Testing

Run the included test script to verify functionality:

```bash
python components/extractors/path_extractor/test_path_extractor.py
```

## Best Practices

1. **Always Apply First**: Apply PathExtractor before other extractors in your pipeline
2. **Set Source Field**: Always set `doc.source` after parsing and before extraction
3. **Use Full Paths**: Store full paths unless you have specific storage constraints
4. **Normalize for Web**: Enable `normalize_paths` for web applications
5. **Test Retrieval**: Verify path information is preserved after vector database storage

## Contributing

When modifying the Path Extractor:

1. Update the schema.json if adding new configuration options
2. Add new defaults to defaults.json
3. Update the documentation in path_extractor.md
4. Add tests for new functionality
5. Ensure backward compatibility

## License

Part of the RAG system - see main project license.