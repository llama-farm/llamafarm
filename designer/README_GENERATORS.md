# TypeScript Type Generator for Designer UI

This directory contains the unified type generator that creates TypeScript types and constants for the Designer UI from `rag/schema.yaml`.

## What It Generates

The generator creates two TypeScript files in `src/components/Rag/generated/`:

### 1. `ragTypes.ts` - Parser and Extractor Types
- Parser type constants (`PARSER_TYPES`)
- Extractor type constants (`EXTRACTOR_TYPES`)
- Default configuration functions (`getDefaultParserConfig`, `getDefaultExtractorConfig`)
- Schema metadata with properties and descriptions
- TypeScript type definitions

### 2. `databaseTypes.ts` - Database, Embedder, and Retrieval Strategy Types
- Vector store/database type constants (`VECTOR_STORE_TYPES`)
- Embedder type constants (`EMBEDDER_TYPES`)
- Retrieval strategy type constants (`RETRIEVAL_STRATEGY_TYPES`)
- Default configuration functions for each
- Schema metadata with categorization
- Helper functions for filtering by category/complexity

## Key Features

### Zero Hardcoding - Fully Schema-Driven
All type mappings, defaults, and configurations are **automatically derived from the schema structure**. No hardcoded dictionaries or magic strings!

#### Parser Type Discovery
Analyzes `definitions.parsers.*Config` keys to derive parser types:
- `pdfParserPyPDF2Config` → `PDFParser_PyPDF2`
- `csvParserPandasConfig` → `CSVParser_Pandas`
- `autoParserConfig` → `auto`

#### Extractor Type Discovery
Uses multiple strategies to map extractor types to config keys:
1. Direct match: `KeywordExtractor` → `keywordExtractorConfig`
2. Remove "Extractor" suffix: `SummaryExtractor` → `summaryExtractorConfig`
3. Algorithm enum lookup: `RAKEExtractor`, `TFIDFExtractor`, `YAKEExtractor` → `keywordExtractorConfig`

#### Vector Store Type Discovery
Maps store types to config keys:
- `ChromaStore` → `chromaStoreConfig`
- `FAISSStore` → `faissStoreConfig`

#### Embedder Type Discovery
Handles special cases automatically:
- `OllamaEmbedder` → `ollamaEmbedderConfig`
- `HuggingFaceEmbedder` → `huggingfaceEmbedderConfig`
- `SentenceTransformerEmbedder` → `sentenceTransformerConfig`

#### Retrieval Strategy Type Discovery
Maps strategy types by removing "Strategy" suffix:
- `BasicSimilarityStrategy` → `basicSimilarityConfig`
- `HybridUniversalStrategy` → `hybridUniversalConfig`

### Fully Extensible
Adding new parsers, extractors, stores, embedders, or strategies to the schema automatically includes them in the generated UI types. No code changes needed!

**Example:** Add a new parser to `rag/schema.yaml`:
```yaml
definitions:
  parsers:
    excelParserPandasConfig:
      type: object
      title: "Excel Parser (Pandas)"
      properties:
        sheet_name:
          type: string
          default: "Sheet1"
```

Run `./generate-types.sh` and the new `EXCELParser_Pandas` type is automatically available in the UI!

## Usage

### Generate Types

```bash
cd designer
./generate-types.sh
```

Or from the `rag/` directory for backward compatibility:

```bash
cd rag
./generate-ui-types.sh         # Generates ragTypes.ts only
./generate-db-embedding-types.sh # Generates databaseTypes.ts only
```

### Import Generated Types

```typescript
// Parser and Extractor types
import {
  PARSER_TYPES,
  ParserType,
  getDefaultParserConfig,
  PARSER_SCHEMAS,
  EXTRACTOR_TYPES,
  ExtractorType,
  getDefaultExtractorConfig,
  EXTRACTOR_SCHEMAS,
} from '@/components/Rag/generated/ragTypes'

// Database, Embedder, and Retrieval Strategy types
import {
  VECTOR_STORE_TYPES,
  VectorStoreType,
  getDefaultVectorStoreConfig,
  EMBEDDER_TYPES,
  EmbedderType,
  getDefaultEmbedderConfig,
  RETRIEVAL_STRATEGY_TYPES,
  RetrievalStrategyType,
  getDefaultRetrievalStrategyConfig,
  getVectorStoresByCategory,
  getEmbeddersByCategory,
  getRetrievalStrategiesByComplexity,
} from '@/components/Rag/generated/databaseTypes'
```

### Example: Creating a Parser with Schema Defaults

```typescript
import { getDefaultParserConfig, PARSER_SCHEMAS } from '@/components/Rag/generated/ragTypes'

const parserType: ParserType = 'PDFParser_LlamaIndex'
const defaultConfig = getDefaultParserConfig(parserType)
const schema = PARSER_SCHEMAS[parserType]

const newParser = {
  type: parserType,
  config: defaultConfig,  // { chunk_size: 1000, chunk_overlap: 200, ... }
  file_include_patterns: schema.defaultExtensions.map(ext => `*${ext}`),  // ['*.pdf']
  priority: 50,
}
```

### Example: Filtering Vector Stores by Category

```typescript
import { getVectorStoresByCategory } from '@/components/Rag/generated/databaseTypes'

const localStores = getVectorStoresByCategory('local')    // ['ChromaStore', 'QdrantStore']
const cloudStores = getVectorStoresByCategory('cloud')    // ['PineconeStore']
const memoryStores = getVectorStoresByCategory('memory')  // ['FAISSStore']
```

## Files

- **`generate-types.py`** - Unified Python generator script
- **`generate-types.sh`** - Shell script wrapper
- **`README_GENERATORS.md`** - This documentation

## How It Works

1. **Load Schema**: Reads `rag/schema.yaml` using PyYAML
2. **Discover Types**: Analyzes schema structure to find all type definitions
3. **Build Mappings**: Creates type-to-config mappings dynamically (no hardcoding!)
4. **Extract Metadata**: Pulls titles, descriptions, defaults, and property definitions
5. **Generate TypeScript**: Creates properly typed constants, functions, and interfaces
6. **Write Files**: Outputs to `designer/src/components/Rag/generated/`

## Schema Requirements

For automatic type discovery to work, the schema should follow these conventions:

### Parsers
```yaml
definitions:
  parsers:
    {format}Parser{Tool}Config:  # e.g., pdfParserPyPDF2Config
      type: object
      title: "Human Readable Title"
      description: "What this parser does"
      properties:
        # ... config properties with defaults
```

### Extractors
```yaml
definitions:
  extractors:
    {name}ExtractorConfig:  # e.g., keywordExtractorConfig
      type: object
      title: "Human Readable Title"
      properties:
        algorithm:  # Optional: for multi-algorithm extractors
          enum: [RAKE, TFIDF, YAKE]
```

### Vector Stores
```yaml
definitions:
  vectorStores:
    {name}StoreConfig:  # e.g., chromaStoreConfig
      type: object
      title: "Human Readable Title"
      properties:
        # ... config properties
```

### Embedders
```yaml
definitions:
  embedders:
    {name}EmbedderConfig:  # e.g., ollamaEmbedderConfig
      type: object
      title: "Human Readable Title"
      properties:
        # ... config properties
```

### Retrieval Strategies
```yaml
definitions:
  retrievalStrategies:
    {name}Config:  # e.g., basicSimilarityConfig
      type: object
      title: "Human Readable Title"
      properties:
        # ... config properties
```

## Backward Compatibility

The old generator scripts in `rag/` are maintained for backward compatibility but are now simple wrappers around the unified generator:

- `rag/generate-ui-types.sh` → Calls unified generator (generates ragTypes.ts)
- `rag/generate-db-embedding-types.sh` → Calls unified generator (generates databaseTypes.ts)

**Recommended:** Use `designer/generate-types.sh` to generate both files at once.

## Benefits

### For Developers
- **Single Source of Truth**: Schema drives everything
- **Type Safety**: Full TypeScript types for all RAG components
- **Auto-completion**: IDE suggestions for all types and configs
- **Default Values**: Schema defaults automatically applied

### For Maintainers
- **No Hardcoding**: All mappings derived from schema structure
- **Easy Extension**: Add to schema, run generator, done!
- **Consistent**: Same pattern across all component types
- **Self-Documenting**: Schema serves as documentation

### For Users
- **Better UI**: Type-safe forms with proper defaults
- **Fewer Errors**: Valid configurations by default
- **Discoverable**: All available options shown in UI

## Troubleshooting

### Import Error: No module named 'yaml'

The generator requires the PyYAML dependency. Run from the `server/` directory where `pyproject.toml` is:

```bash
cd server
uv run python ../designer/generate-types.py
```

Or use the provided shell script which handles this automatically:

```bash
cd designer
./generate-types.sh
```

### Generated Types Don't Include My New Parser

Check that your parser config follows the naming convention:
- Must be in `definitions.parsers`
- Must end with `Config`
- Must contain "Parser" in the name (e.g., `pdfParserToolConfig`)

### Type Mapping Not Working

Enable debug output by modifying `generate-types.py`:

```python
# Add after building the mapping
print(f"Parser type mapping: {build_parser_type_mapping(schema)}")
```

This will show exactly how types are being mapped to config keys.

## Migration from Old Generators

If you were using the old separate generators:

**Before:**
```bash
cd rag
./generate-ui-types.sh           # Parsers/Extractors
./generate-db-embedding-types.sh  # Stores/Embedders/Strategies
```

**After:**
```bash
cd designer
./generate-types.sh  # Everything!
```

The old scripts still work but now call the unified generator internally.

## Future Enhancements

Potential improvements:
- [ ] Add validation to ensure schema follows conventions
- [ ] Generate Zod schemas for runtime validation
- [ ] Add CLI flags for selective generation
- [ ] Support for custom categorization rules from schema
- [ ] Generate example configurations for documentation

## Questions?

For issues or questions:
1. Check that `rag/schema.yaml` follows the conventions above
2. Run the generator with debug output
3. Open an issue with the schema snippet and error message
