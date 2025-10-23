# Database & Embedding Type Generator

This directory contains a script to automatically generate TypeScript types and constants for the Database/Embedding UI from the RAG schema.

## Overview

The generator reads `rag/schema.yaml` and produces:
- Vector store/database type constants and TypeScript types
- Embedder type constants and TypeScript types
- Retrieval strategy type constants and TypeScript types
- Default configuration functions for all types
- Schema metadata with categorization
- Helper functions for filtering by category/complexity

## Usage

```bash
cd rag
./generate-db-embedding-types.sh
```

## Output

The script generates:
```
designer/src/components/Rag/generated/databaseTypes.ts
```

This file contains:
- `VECTOR_STORE_TYPES` - Array of all vector store types (ChromaStore, QdrantStore, etc.)
- `EMBEDDER_TYPES` - Array of all embedder types (OllamaEmbedder, OpenAIEmbedder, etc.)
- `RETRIEVAL_STRATEGY_TYPES` - Array of all retrieval strategy types
- `getDefaultVectorStoreConfig(type)` - Get default config for a vector store
- `getDefaultEmbedderConfig(type)` - Get default config for an embedder
- `getDefaultRetrievalStrategyConfig(type)` - Get default config for a retrieval strategy
- `VECTOR_STORE_SCHEMAS` - Metadata with categorization (local/cloud/memory)
- `EMBEDDER_SCHEMAS` - Metadata with categorization (local/cloud/huggingface)
- `RETRIEVAL_STRATEGY_SCHEMAS` - Metadata with complexity levels (basic/intermediate/advanced)
- Helper functions for filtering by category/complexity

## When to Run

Run this script whenever you:
1. Add a new vector store type to `rag/schema.yaml`
2. Add a new embedder type to `rag/schema.yaml`
3. Add a new retrieval strategy type to `rag/schema.yaml`
4. Change default configuration values
5. Update vector store/embedder/retrieval descriptions or metadata

## Example Usage in UI

```typescript
import {
  VECTOR_STORE_TYPES,
  EMBEDDER_TYPES,
  RETRIEVAL_STRATEGY_TYPES,
  getDefaultVectorStoreConfig,
  getDefaultEmbedderConfig,
  getDefaultRetrievalStrategyConfig,
  VECTOR_STORE_SCHEMAS,
  EMBEDDER_SCHEMAS,
  RETRIEVAL_STRATEGY_SCHEMAS,
  getVectorStoresByCategory,
  getEmbeddersByCategory,
  getRetrievalStrategiesByComplexity
} from '@/components/Rag/generated/databaseTypes'

// Get all available vector stores
const stores = VECTOR_STORE_TYPES
// ["ChromaStore", "QdrantStore", "FAISSStore", "PineconeStore"]

// Get default config for ChromaDB
const chromaConfig = getDefaultVectorStoreConfig("ChromaStore")
// Returns: { collection_name: "documents", port: 8000, distance_function: "cosine", ... }

// Get default config for Ollama embedder
const ollamaConfig = getDefaultEmbedderConfig("OllamaEmbedder")
// Returns: { model: "nomic-embed-text", base_url: "http://localhost:11434", ... }

// Get schema metadata
const chromaSchema = VECTOR_STORE_SCHEMAS["ChromaStore"]
// Returns: { type: "ChromaStore", title: "Chroma Store Configuration", category: "local", ... }

// Filter stores by category
const localStores = getVectorStoresByCategory("local")
// Returns: ["ChromaStore", "QdrantStore"]

const cloudStores = getVectorStoresByCategory("cloud")
// Returns: ["PineconeStore"]

// Filter embedders by category
const localEmbedders = getEmbeddersByCategory("local")
// Returns: ["OllamaEmbedder"]

const cloudEmbedders = getEmbeddersByCategory("cloud")
// Returns: ["OpenAIEmbedder"]

// Filter retrieval strategies by complexity
const basicStrategies = getRetrievalStrategiesByComplexity("basic")
// Returns: ["BasicSimilarityStrategy", "VectorRetriever", ...]

const advancedStrategies = getRetrievalStrategiesByComplexity("advanced")
// Returns: ["HybridUniversalStrategy", "MultiQueryStrategy"]
```

## Generated Types

### Vector Stores (4 types)
- **ChromaStore** - Local vector database (category: local)
- **QdrantStore** - Local/cloud vector database (category: local)
- **FAISSStore** - In-memory vector index (category: memory)
- **PineconeStore** - Cloud vector database (category: cloud)

### Embedders (4 types)
- **OllamaEmbedder** - Local embedding via Ollama (category: local)
- **HuggingFaceEmbedder** - HuggingFace transformers (category: huggingface)
- **OpenAIEmbedder** - OpenAI embedding API (category: cloud)
- **SentenceTransformerEmbedder** - Sentence transformers (category: huggingface)

### Retrieval Strategies (11 types)
- **BasicSimilarityStrategy** - Simple similarity search (complexity: basic)
- **VectorRetriever** - Vector-based retrieval (complexity: basic)
- **BM25Retriever** - BM25 algorithm (complexity: basic)
- **MetadataFilteredStrategy** - Filter by metadata (complexity: intermediate)
- **RerankedStrategy** - Rerank results (complexity: intermediate)
- **MultiQueryStrategy** - Multiple query variations (complexity: advanced)
- **HybridUniversalStrategy** - Combine multiple strategies (complexity: advanced)
- And more...

## Files

- `generate-db-embedding-types.sh` - Shell wrapper script (run this)
- `generate-db-embedding-types.py` - Python generator (called by the shell script)
- `schema.yaml` - Source schema (single source of truth)

## Generated File

The generated file is **auto-generated** and should:
- ✅ Be committed to git (so UI devs don't need to run the generator)
- ⚠️  Never be manually edited (changes will be overwritten)
- 📝 Be regenerated whenever the schema changes

## Workflow

1. Developer updates `rag/schema.yaml` with new vector store/embedder/retrieval type
2. Developer runs `cd rag && ./generate-db-embedding-types.sh`
3. Generated TypeScript file is updated
4. Developer commits both schema.yaml and generated databaseTypes.ts
5. UI automatically picks up new types/configs/helpers

## Companion Generators

This generator is part of a set of schema-driven generators:

1. **generate-types.sh** (config/) - Generates Python/Go types from config schema
2. **generate-ui-types.sh** (rag/) - Generates parser/extractor types for Data UI
3. **generate-db-embedding-types.sh** (rag/) - Generates database/embedding types for Databases UI ← YOU ARE HERE

All follow the same pattern: **Schema → Generator → Types → Commit**
