# RAG UI Type Generator

This directory contains a script to automatically generate TypeScript types and constants for the Designer UI from the RAG schema.

## Overview

The generator reads `rag/schema.yaml` and produces:
- Parser type constants and TypeScript types
- Extractor type constants and TypeScript types
- Default configuration functions for all parsers/extractors
- Schema metadata (titles, descriptions, default file extensions)

## Usage

```bash
cd rag
./generate-ui-types.sh
```

## Output

The script generates:
```
designer/src/components/Rag/generated/ragTypes.ts
```

This file contains:
- `PARSER_TYPES` - Array of all available parser types
- `EXTRACTOR_TYPES` - Array of all available extractor types
- `getDefaultParserConfig(type)` - Function to get default config for a parser
- `getDefaultExtractorConfig(type)` - Function to get default config for an extractor
- `PARSER_SCHEMAS` - Metadata about each parser (title, description, extensions)
- `EXTRACTOR_SCHEMAS` - Metadata about each extractor (title, description)

## When to Run

Run this script whenever you:
1. Add a new parser type to `rag/schema.yaml`
2. Add a new extractor type to `rag/schema.yaml`
3. Change default configuration values
4. Update parser/extractor descriptions or metadata

## Example Usage in UI

```typescript
import {
  PARSER_TYPES,
  EXTRACTOR_TYPES,
  getDefaultParserConfig,
  getDefaultExtractorConfig,
  PARSER_SCHEMAS,
  EXTRACTOR_SCHEMAS
} from '@/components/Rag/generated/ragTypes'

// Get all available parsers
const parsers = PARSER_TYPES // ["PDFParser_PyPDF2", "PDFParser_LlamaIndex", ...]

// Get default config for a parser
const pdfConfig = getDefaultParserConfig("PDFParser_PyPDF2")
// Returns: { chunk_size: 1000, chunk_overlap: 100, ... }

// Get schema metadata
const pdfSchema = PARSER_SCHEMAS["PDFParser_PyPDF2"]
// Returns: { type: "PDFParser_PyPDF2", title: "PDF Parser (PyPDF2) Configuration", ... }
```

## Files

- `generate-ui-types.sh` - Shell wrapper script (run this)
- `generate-ui-types.py` - Python generator (called by the shell script)
- `schema.yaml` - Source schema (single source of truth)

## Generated File

The generated file is **auto-generated** and should:
- ✅ Be committed to git (so UI devs don't need to run the generator)
- ⚠️  Never be manually edited (changes will be overwritten)
- 📝 Be regenerated whenever the schema changes

## Workflow

1. Developer updates `rag/schema.yaml` with new parser/extractor
2. Developer runs `cd rag && ./generate-ui-types.sh`
3. Generated TypeScript file is updated
4. Developer commits both schema.yaml and generated ragTypes.ts
5. UI automatically picks up new types/configs

## Similar to

This follows the same pattern as `config/generate-types.sh` which generates:
- Python Pydantic models from config schema
- Go structs for CLI from config schema

Both use the same principle: **Schema is the single source of truth, types are generated.**
