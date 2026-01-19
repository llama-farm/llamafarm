# LlamaFarm RAG Migration Guide

This guide helps you migrate between different RAG configurations.

## Migrating to Lightweight Stack

If you're currently using the full RAG package and want to switch to the lightweight stack built into the server:

### Step 1: Update Configuration

Change your `rag` configuration from advanced to universal:

**Before (Advanced):**
```yaml
rag:
  parser:
    type: "DoclingParser"
    # ... complex config
```

**After (Universal):**
```yaml
rag:
  parser:
    type: "universal"
    chunk_size: 512
    chunk_strategy: "semantic"
    use_ocr: true
```

### Step 2: Update Dependencies

The lightweight stack is built into the server, so you don't need the full RAG package dependencies.

**Server dependencies include:**
- markitdown>=0.1.4
- semchunk>=2.0.0
- tiktoken>=0.5.0
- yake>=0.4.8
- chromadb>=1.0.0

**You can remove (if not needed elsewhere):**
- llama-index
- docling
- sentence-transformers (unless using cross-encoder reranking)

### Step 3: Update API Endpoints

The server-integrated RAG uses different endpoints:

**Advanced RAG (separate service):**
- `POST /v1/rag/ingest` → RAG worker
- `POST /v1/rag/query` → RAG worker

**Lightweight RAG (server-integrated):**
- `POST /v1/rag-lite/ingest` → Main server
- `GET /v1/rag-lite/search` → Main server

### Step 4: Verify Operation

```bash
# Check server RAG stats
curl http://localhost:8000/v1/rag-lite/stats

# Test search
curl "http://localhost:8000/v1/rag-lite/search?query=test&top_k=5"
```

## Migrating to Advanced Stack

If you need more features and want to switch from lightweight to advanced:

### Step 1: Install RAG Package

```bash
# Install full RAG dependencies
uv add "llama-rag[all]"
```

### Step 2: Start RAG Worker

```bash
# Start the RAG Celery worker
nx start rag
```

### Step 3: Update Configuration

Use the `rag_advanced.yaml` template:

```yaml
rag:
  parser:
    type: "DoclingParser"
    chunk_size: 1000
    extract_tables: true
    extract_headings: true
```

### Step 4: Migrate Data

If you have existing documents in ChromaDB:

```python
# Export from lightweight
from server.services.rag import RAGService
rag = RAGService()
docs = rag.get_all_documents()

# Import to advanced RAG
# Use the RAG API endpoints
```

## Feature Comparison

| Feature | Lightweight (Server) | Advanced (RAG Package) |
|---------|---------------------|------------------------|
| Basic PDF parsing | ✅ | ✅ |
| Table extraction | ❌ | ✅ |
| Layout analysis | ❌ | ✅ |
| OCR | ✅ (via Runtime) | ✅ (integrated) |
| Semantic chunking | ✅ | ✅ |
| Keyword extraction | ✅ (YAKE) | ✅ (multiple) |
| Entity extraction | ✅ (GLiNER opt.) | ✅ (spaCy) |
| ChromaDB | ✅ (embedded) | ✅ (server) |
| Qdrant/Milvus | ❌ | ✅ |
| Reranking | ❌ | ✅ |
| Multi-query | ❌ | ✅ |

## Coexistence Mode

You can run both stacks simultaneously:

1. **Lightweight** for simple, quick ingestion
2. **Advanced** for complex documents

Configure routing in your application to choose the appropriate stack based on document type.

## Troubleshooting

### Missing Dependencies

```bash
# For lightweight stack
uv add markitdown[all] semchunk tiktoken yake

# For advanced stack
uv add "llama-rag[all]"
```

### ChromaDB Conflicts

If both stacks try to use the same ChromaDB:

```yaml
# Lightweight
vector_store:
  persist_directory: "./chroma_lite"

# Advanced
vector_store:
  persist_directory: "./chroma_advanced"
```

### Import Errors

Make sure your PYTHONPATH includes the correct directories:

```bash
# For server (lightweight)
export PYTHONPATH="${PYTHONPATH}:./server"

# For RAG package
export PYTHONPATH="${PYTHONPATH}:./rag"
```
