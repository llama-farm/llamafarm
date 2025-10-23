# RAG Schema Generators

This directory contains automated type generators that create TypeScript types and constants from the RAG schema for use in the Designer UI.

## Philosophy

**Schema is the single source of truth.** All types, configurations, and metadata are generated from `rag/schema.yaml`.

This follows the same pattern as `config/generate-types.sh` which generates Python and Go types from the config schema.

## Available Generators

### 1. Parser & Extractor Generator

```bash
./generate-ui-types.sh
```

**Generates:** `designer/src/components/Rag/generated/ragTypes.ts`

**Contents:**
- 16 parser types (PDF, CSV, Excel, DOCX, Markdown, Text, MSG)
- 13 extractor types (Keywords, Entities, DateTime, Headings, etc.)
- Default configurations for all parsers and extractors
- Parser/extractor metadata (titles, descriptions, file extensions)

**Use case:** Data page UI for managing processing strategies

**Read more:** [README_GENERATE_UI_TYPES.md](./README_GENERATE_UI_TYPES.md)

---

### 2. Database & Embedding Generator

```bash
./generate-db-embedding-types.sh
```

**Generates:** `designer/src/components/Rag/generated/databaseTypes.ts`

**Contents:**
- 4 vector store types (Chroma, Qdrant, FAISS, Pinecone)
- 4 embedder types (Ollama, HuggingFace, OpenAI, SentenceTransformer)
- 11 retrieval strategy types (Basic, Filtered, MultiQuery, Reranked, Hybrid)
- Default configurations for all types
- Categorized metadata (local/cloud/memory for stores, basic/intermediate/advanced for strategies)
- Helper functions for filtering by category/complexity

**Use case:** Databases page UI for managing vector stores and embedding strategies

**Read more:** [README_GENERATE_DB_TYPES.md](./README_GENERATE_DB_TYPES.md)

---

## Quick Reference

| Generator | Command | Output | Types Generated |
|-----------|---------|--------|-----------------|
| **Parser/Extractor** | `./generate-ui-types.sh` | `ragTypes.ts` | 16 parsers, 13 extractors |
| **Database/Embedding** | `./generate-db-embedding-types.sh` | `databaseTypes.ts` | 4 stores, 4 embedders, 11 retrievers |

## When to Run

Run the appropriate generator whenever you:
- ✅ Add a new parser/extractor/embedder/store type to `rag/schema.yaml`
- ✅ Change default configuration values in the schema
- ✅ Update descriptions or metadata in the schema
- ✅ Add new schema properties that should be exposed to the UI

## Workflow

```bash
# 1. Update RAG schema
vim rag/schema.yaml

# 2. Run appropriate generator(s)
./generate-ui-types.sh          # For parser/extractor changes
./generate-db-embedding-types.sh  # For database/embedding changes

# 3. Review generated files
git diff designer/src/components/Rag/generated/

# 4. Commit both schema and generated files
git add rag/schema.yaml designer/src/components/Rag/generated/
git commit -m "feat: add new parser type"
```

## Generated Files

All generated files are:
- ✅ **Committed to git** - UI developers don't need Python/uv installed
- ⚠️  **Never manually edited** - Changes will be overwritten on next generation
- 📝 **Regenerated on schema changes** - Always in sync with schema
- 🔒 **Type-safe** - TypeScript const arrays and literal types

## Benefits

### For Schema Developers
- Update schema once, UI types update automatically
- No manual TypeScript type maintenance
- Guaranteed consistency between backend and frontend

### For UI Developers
- Import types directly from generated files
- Auto-complete for all parser/extractor/embedder types
- Default configs always match schema
- Categorization helpers for UI organization

### For Everyone
- Single source of truth (schema.yaml)
- No type mismatches between backend and frontend
- Clear separation of concerns
- Easy to extend with new types

## Example Usage

### Parser/Extractor Types
```typescript
import {
  PARSER_TYPES,
  getDefaultParserConfig,
  PARSER_SCHEMAS
} from '@/components/Rag/generated/ragTypes'

// Dropdown of all parsers
<Select>
  {PARSER_TYPES.map(type => (
    <Option value={type}>{PARSER_SCHEMAS[type].title}</Option>
  ))}
</Select>

// Get default config when user adds new parser
const newParser = {
  type: "PDFParser_PyPDF2",
  config: getDefaultParserConfig("PDFParser_PyPDF2"),
  file_include_patterns: PARSER_SCHEMAS["PDFParser_PyPDF2"].defaultExtensions
}
```

### Database/Embedding Types
```typescript
import {
  EMBEDDER_TYPES,
  getDefaultEmbedderConfig,
  getEmbeddersByCategory
} from '@/components/Rag/generated/databaseTypes'

// Show only local embedders
const localEmbedders = getEmbeddersByCategory("local")

// Get default Ollama config
const ollamaConfig = getDefaultEmbedderConfig("OllamaEmbedder")
```

## Architecture

```
rag/schema.yaml
    ↓
[Python Generator Script]
    ↓
designer/src/components/Rag/generated/*.ts
    ↓
[React UI Components]
```

All generators follow this pattern:
1. Parse YAML schema
2. Extract type definitions and defaults
3. Generate TypeScript with type safety
4. Write to `designer/src/components/Rag/generated/`

## Related Generators

### Config Types (config/)
- **Script:** `config/generate-types.sh`
- **Output:** `config/datamodel.py` (Pydantic), `cli/cmd/config/types.go` (Go structs)
- **Purpose:** Backend type safety for config validation

### RAG UI Types (rag/) ← YOU ARE HERE
- **Scripts:** `generate-ui-types.sh`, `generate-db-embedding-types.sh`
- **Output:** `designer/src/components/Rag/generated/*.ts`
- **Purpose:** Frontend type safety for RAG UI

## Troubleshooting

### Generator fails with Python error
```bash
# Ensure you have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run from rag directory
cd rag
./generate-ui-types.sh
```

### Generated types don't match schema
```bash
# Ensure schema.yaml is valid YAML
yamllint schema.yaml

# Regenerate
./generate-ui-types.sh
./generate-db-embedding-types.sh
```

### UI can't import generated types
```bash
# Ensure TypeScript can resolve the path
# Check tsconfig.json has proper path mapping for @/components

# Verify files were generated
ls -la designer/src/components/Rag/generated/
```

## Contributing

When adding new types to the schema:
1. Add the type definition to `rag/schema.yaml`
2. Add default values for all config properties
3. Run the appropriate generator
4. Update UI components to use the new type
5. Test that defaults work correctly
6. Commit schema + generated files together

## Future Enhancements

Potential additions:
- [ ] Generate Zod/Yup validation schemas for forms
- [ ] Generate React form components from schema
- [ ] Generate API client types
- [ ] Generate documentation from schema
- [ ] Add schema version tracking
