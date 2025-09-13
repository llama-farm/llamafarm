# RAG Pipeline with Separate Fields
## Successful Implementation of Separate data_processing_strategy and database

**Date:** September 12, 2025  
**Status:** ✅ WORKING

---

## Architecture

The refactored system now properly separates:
1. **data_processing_strategy** - How to parse and extract from files
2. **database** - Where to store the processed documents

This is much cleaner than combining them into a single strategy name!

---

## Configuration Structure

### Data Processing Strategy
```yaml
data_processing_strategies:
  - name: universal_processor
    description: "Single strategy handling all document types"
    parsers:
      - type: PDFParser_LlamaIndex
        file_include_patterns: ["*.pdf"]
        priority: 100
        config: {...}
    extractors:
      - type: TableExtractor
        file_include_patterns: ["*.pdf"]
        priority: 100
        config: {...}
```

### Database Configuration
```yaml
databases:
  - name: main_database
    type: ChromaStore
    config:
      persist_directory: ./data/chroma_db
      collection_name: documents
    embedding_strategies:
      - name: default_embeddings
        type: OllamaEmbedder
        config: {...}
    retrieval_strategies:
      - name: basic_search
        type: BasicSimilarityStrategy
        config: {...}
```

---

## Usage Example

### Initialize with Separate Fields
```python
from rag.core.ingest_handler import IngestHandler

handler = IngestHandler(
    config_path="llamafarm.yaml",
    data_processing_strategy="universal_processor",  # Separate field
    database="main_database"                         # Separate field
)
```

### Ingest a File
```python
# Read file
with open("document.pdf", 'rb') as f:
    file_data = f.read()

metadata = {
    'filename': 'document.pdf',
    'filepath': '/path/to/document.pdf',
    'size': len(file_data)
}

# Ingest (strategy and database already set in handler)
result = handler.ingest_file(
    file_data=file_data,
    metadata=metadata
)
```

---

## Test Results

### Test File: `test_report.txt`
```
Testing with:
  Data Processing Strategy: universal_processor
  Database: main_database

Ingestion Result:
  Status: success
  Documents processed: 1
  Parsers used: ['TextParser_Python']
  Extractors applied: ['ContentStatisticsExtractor', 'EntityExtractor', 'KeywordExtractor']
```

---

## Schema Updates

Added to `/rag/schema.yaml`:

### Parser Fields
```yaml
file_include_patterns:
  type: array
  items:
    type: string
  description: Glob patterns for files to include
file_exclude_patterns:
  type: array
  items:
    type: string
  description: Glob patterns for files to exclude
priority:
  type: integer
  description: Parser priority (higher = try first)
```

### Extractor Fields
```yaml
file_include_patterns:
  type: array
  items:
    type: string
  description: Glob patterns for files to apply extractor
file_exclude_patterns:
  type: array
  items:
    type: string
  description: Glob patterns for files to exclude
priority:
  type: integer
  description: Extractor priority (higher = apply first)
```

---

## Benefits of Separate Fields

1. **Clarity** - Clear separation of concerns
2. **Flexibility** - Can mix and match strategies with databases
3. **Reusability** - Same processing strategy can be used with different databases
4. **Maintainability** - Easier to understand and modify
5. **API Design** - Cleaner API with explicit parameters

---

## CLI Integration (Expected)

```bash
# Create dataset with separate fields
lf datasets add my-dataset \
  --data-processing-strategy universal_processor \
  --database main_database

# Or with defaults
lf datasets add my-dataset  # Uses defaults from config

# Ingest files
lf datasets ingest my-dataset document.pdf
```

---

## Next Steps

1. ✅ Schema updated with pattern fields
2. ✅ IngestHandler accepts separate fields
3. ✅ BlobProcessor works with patterns
4. ✅ Test confirms separate fields work
5. ⏳ CLI needs to pass separate fields to API
6. ⏳ API endpoint needs to accept separate fields
7. ⏳ Connect real embedders and stores (currently mocked)