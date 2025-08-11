# Text Parser

**Framework:** Pure Python

**When to use:** Parse plain text files, logs, and unstructured text documents.

**Schema fields:**
- `chunk_size`: Characters per chunk
- `chunk_overlap`: Overlap between chunks
- `preserve_line_breaks`: Keep original line breaks
- `detect_structure`: Auto-detect lists, headers
- `encoding`: Text encoding (auto-detect if None)

**Best practices:**
- Use overlap to preserve context
- Adjust chunk size for model limits
- Detect structure for better chunking
- Handle encoding for international text