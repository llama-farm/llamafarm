# Path Extractor

**Framework:** Pure Python (pathlib)

**When to use:** Preserve source file path information in document metadata for retrieval systems, search result display, and document traceability.

## Overview

The Path Extractor ensures that source file information is reliably preserved throughout the RAG pipeline. This is critical for:
- Displaying source files in search results
- Tracing documents back to their origins
- Managing multi-file document collections
- Creating audit trails

## Schema Fields

- `store_full_path`: Store the complete file path (default: true)
- `store_filename`: Store just the filename (default: true)
- `store_directory`: Store the parent directory path (default: true)
- `store_extension`: Store the file extension (default: true)
- `normalize_paths`: Convert to forward slashes for cross-platform compatibility (default: false)
- `relative_to`: Make paths relative to specified directory (default: null)

## Best Practices

1. **Always Apply First**: Apply PathExtractor before other extractors to ensure source information is available
2. **Use Full Paths**: Enable `store_full_path` for complete traceability
3. **Normalize for Web**: Enable `normalize_paths` when building web applications
4. **Relative for Portability**: Use `relative_to` when building portable document collections

## Common Use Cases

### Search Result Display
```python
path_extractor = PathExtractor("path_extractor", {
    "store_full_path": true,
    "store_filename": true,
    "store_directory": true,
    "store_extension": true
})
```

### Web Application
```python
path_extractor = PathExtractor("path_extractor", {
    "store_full_path": true,
    "store_filename": true,
    "normalize_paths": true,
    "relative_to": "./uploads"
})
```

### Minimal Storage
```python
path_extractor = PathExtractor("path_extractor", {
    "store_full_path": false,
    "store_filename": true,
    "store_directory": false,
    "store_extension": false
})
```

## Output Structure

The extractor adds path information to document metadata in two ways:

1. **In extractors section**:
```json
{
  "extractors": {
    "path": {
      "full_path": "/path/to/document.pdf",
      "filename": "document.pdf",
      "directory": "/path/to",
      "extension": ".pdf"
    }
  }
}
```

2. **As top-level fields** (for compatibility with search displays):
```json
{
  "file_path": "/path/to/document.pdf",
  "file_name": "document.pdf",
  "directory": "/path/to",
  "file_extension": ".pdf"
}
```

## Integration Example

```python
from components.extractors.path_extractor import PathExtractor

# Initialize extractor
path_extractor = PathExtractor("path_extractor", {
    "store_full_path": true,
    "store_filename": true,
    "store_directory": true,
    "store_extension": true
})

# Apply to documents (after parsing, before other extractors)
documents = parser.parse(file_path)
for doc in documents:
    doc.source = file_path  # Ensure source is set

documents = path_extractor.extract(documents)
```

## Troubleshooting

### Source showing as "unknown" in search results
- Ensure documents have `source` field set before extraction
- Apply PathExtractor before storing in vector database
- Check that vector store preserves metadata fields

### Path not showing in specific UI
- Some UIs look for `file_name` field specifically
- PathExtractor stores both `filename` and `file_name` for compatibility

### Cross-platform path issues
- Enable `normalize_paths` to convert all paths to forward slashes
- Use `relative_to` to make paths portable across systems