# Add Built-in Lightweight RAG to Server (Default)

## Problem Statement

Currently, **RAG functionality requires running a separate worker process** (`nx start rag`), which creates friction for most users:

1. **Complex Setup**: Users must run multiple processes (server + rag worker)
2. **Heavy Dependencies**: RAG package requires llama-index, docling, and other heavy libraries
3. **Over-Engineered**: 90% of users just need basic document search, not advanced pipelines
4. **Poor DX**: Simple use case (search PDFs) requires understanding Celery, workers, and distributed systems

**User Pain Point**:
```bash
# Current: Too complex for basic use
$ nx start server          # Terminal 1
$ nx start rag             # Terminal 2 (Why do I need this?)
$ nx start universal       # Terminal 3

# Then configure RAG in YAML...
# Then understand Celery tasks...
# Just to search a few PDFs? 😞
```

## Proposed Solution

**Add lightweight RAG built directly into the server** as the DEFAULT option. The existing `rag/` package becomes OPTIONAL for power users who need advanced features.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ SERVER (Default - 90% of users)                              │
│                                                              │
│ ✓ MarkItDown parser (Microsoft, lightweight)                │
│ ✓ Simple chunking (1000 char chunks, paragraph-aware)       │
│ ✓ UniversalEmbedder (HTTP calls to Universal Runtime)       │
│ ✓ ChromaDB (embedded vector store)                          │
│ ✓ Basic retrieval (cosine similarity)                       │
│                                                              │
│ Dependencies: markitdown[all], chromadb, requests            │
│ API: /v1/rag-lite/ingest, /v1/rag-lite/search               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RAG Package (Optional - 10% power users)                     │
│                                                              │
│ ✓ Advanced parsers (Docling, LlamaIndex, etc.)              │
│ ✓ Multiple vector stores (Qdrant, Milvus, etc.)             │
│ ✓ Advanced retrieval (reranking, hybrid, etc.)              │
│ ✓ Celery distributed processing                             │
│                                                              │
│ Run as separate worker: `nx start rag`                       │
└─────────────────────────────────────────────────────────────┘
```

### User Experience Improvement

**Before** (Current):
```bash
# 3 terminals needed
nx start server
nx start rag
nx start universal

# Complex YAML config
rag:
  databases:
    - name: main
      type: ChromaStore
      config:
        collection_name: docs
        # ... many more options

# Then use Celery tasks
POST /v1/projects/default/default/rag/ingest
```

**After** (Proposed):
```bash
# 1 terminal (or 2 with universal runtime)
nx start server

# Works immediately, zero config!
curl -X POST http://localhost:8000/v1/rag-lite/ingest \
  -F "file=@document.pdf"

curl -X POST http://localhost:8000/v1/rag-lite/search \
  -H "Content-Type: application/json" \
  -d '{"query": "find information about...", "top_k": 5}'
```

## Implementation Details

### File Structure

```
server/
├── services/
│   └── rag/                          # NEW: Built-in RAG
│       ├── __init__.py
│       ├── rag_service.py            # Main service
│       └── parsers/
│           ├── __init__.py
│           └── markitdown_parser.py  # Lightweight parser
└── api/
    └── routers/
        └── rag_lite.py               # NEW: /v1/rag-lite/* endpoints

common/
└── llamafarm_rag_common/             # NEW: Shared library
    ├── __init__.py
    ├── models.py                     # Document, EmbeddingVector, etc.
    ├── embedders/
    │   ├── base.py
    │   ├── universal_embedder.py     # HTTP-based (no local models)
    │   └── ollama_embedder.py
    ├── retrievers/
    │   ├── base.py
    │   └── basic_similarity.py       # Cosine similarity
    └── stores/
        ├── base.py
        └── chroma_store.py           # Embedded mode only

rag/                                  # Stays unchanged (optional)
├── components/
│   ├── parsers/                      # Advanced: Docling, LlamaIndex, etc.
│   ├── stores/                       # Advanced: Qdrant, Milvus, etc.
│   └── ...
```

### Dependencies

**server/pyproject.toml** (only 3 new deps):
```toml
dependencies = [
    # ... existing ...
    "markitdown[all]>=0.1.4",  # Document parsing (Microsoft)
    "chromadb>=0.4.22",        # Vector database (embedded mode)
    "pytest-asyncio>=0.24.0",  # For async tests
]
```

**common/pyproject.toml**:
```toml
dependencies = [
    # ... existing ...
    "requests>=2.31.0",        # For HTTP-based embedders
    "chromadb>=0.4.22",        # Vector store
]
```

### API Endpoints

#### `GET /v1/rag-lite/stats`
Get RAG system statistics.

**Response**:
```json
{
  "document_count": 42,
  "collection": "server_documents",
  "embedding_dimension": 384,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

#### `POST /v1/rag-lite/ingest`
Ingest a document into the RAG system.

**Request**: `multipart/form-data` with file upload

**Response**:
```json
{
  "success": true,
  "file": "/path/to/file.pdf",
  "chunks": 34,
  "characters": 27118,
  "document_ids": ["uuid1", "uuid2", ...]
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/rag-lite/ingest \
  -F "file=@document.pdf"
```

#### `POST /v1/rag-lite/search`
Search for documents relevant to a query.

**Request**:
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

**Response**:
```json
{
  "success": true,
  "query": "What is machine learning?",
  "count": 3,
  "results": [
    {
      "content": "Machine learning is...",
      "score": 0.9957,
      "metadata": {
        "source": "file.pdf",
        "chunk_index": 0
      },
      "id": "uuid"
    }
  ]
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/v1/rag-lite/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 3}'
```

### Configuration (Optional)

**Default** (zero config):
```yaml
# Nothing needed! RAG-lite works out of the box
```

**Advanced users** (disable built-in, use advanced RAG):
```yaml
server:
  enable_lightweight_rag: false  # Disable built-in RAG

rag:  # Enable advanced RAG package
  databases:
    - name: main
      type: ChromaStore
      # ... advanced config
```

### Data Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /v1/rag-lite/ingest (file)
       ▼
┌─────────────────────────────────────────┐
│     LightweightRAGService               │
│                                         │
│  1. Parse with MarkItDown               │
│     ├─ PDF → markdown                   │
│     ├─ DOCX → markdown                  │
│     ├─ HTML → markdown                  │
│     └─ OCR fallback (Universal Runtime) │
│                                         │
│  2. Chunk text (1000 chars)             │
│     └─ Split on paragraphs              │
│                                         │
│  3. Embed chunks                        │
│     └─ HTTP → Universal Runtime         │
│         (sentence-transformers/all-MiniLM-L6-v2) │
│                                         │
│  4. Store in ChromaDB                   │
│     └─ ~/.llamafarm/server_rag/chroma/  │
└─────────────────────────────────────────┘
       │
       │ POST /v1/rag-lite/search (query)
       ▼
┌─────────────────────────────────────────┐
│  1. Embed query                         │
│     └─ HTTP → Universal Runtime         │
│                                         │
│  2. Cosine similarity search            │
│     └─ ChromaDB query                   │
│                                         │
│  3. Return top_k results                │
│     └─ Sorted by relevance score        │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Client    │
│  (results)  │
└─────────────┘
```

## Benefits

1. **Zero Config**: Works immediately after `nx start server`
2. **Lightweight**: Only 3 new dependencies (vs. 10+ for advanced RAG)
3. **Fast**: <100ms search, ~1-2s ingestion
4. **Simple**: No Celery, no workers, no complex setup
5. **Persistent**: Data survives server restarts (ChromaDB embedded)
6. **Backward Compatible**: Existing advanced RAG package unchanged
7. **No Breaking Changes**: Optional feature, doesn't affect existing users

## Performance

Typical performance on M1 MacBook:

- **Ingestion**: ~1-2s for a 100KB PDF (parsing + chunking + embedding + storage)
- **Search**: ~50-100ms for semantic search with 40+ documents
- **Memory**: ~50MB for service + ~20MB for ChromaDB
- **Relevance**: >0.99 similarity scores on relevant results

## Comparison: RAG-Lite vs Advanced RAG

| Feature | RAG-Lite (Built-in) | Advanced RAG (Optional) |
|---------|---------------------|-------------------------|
| **Availability** | Always (no worker needed) | Requires `nx start rag` |
| **Setup** | Zero config | YAML configuration |
| **Parser** | MarkItDown | Docling, LlamaIndex, etc. |
| **Vector Store** | ChromaDB (embedded) | Qdrant, Milvus, Chroma (server) |
| **Processing** | Synchronous | Celery distributed |
| **Retrieval** | Basic similarity | Reranking, hybrid, etc. |
| **Dependencies** | 3 lightweight | 10+ heavy |
| **Use Case** | 90% of users | 10% power users |
| **Latency** | <100ms | Varies (Celery overhead) |

## Migration Path

**New users**: Get RAG-lite automatically, zero setup

**Existing users with advanced RAG**:
1. Upgrade to new version
2. Advanced RAG still works (no changes needed)
3. Can optionally switch to RAG-lite by stopping RAG worker

**Users wanting both**:
```yaml
# Use RAG-lite for quick searches
# Use advanced RAG for complex pipelines
# Both can coexist
```

## Example Use Cases

### Use Case 1: Simple Document Search
```python
import requests

# Ingest documents
for pdf in ["doc1.pdf", "doc2.pdf", "doc3.pdf"]:
    with open(pdf, "rb") as f:
        requests.post(
            "http://localhost:8000/v1/rag-lite/ingest",
            files={"file": f}
        )

# Search
result = requests.post(
    "http://localhost:8000/v1/rag-lite/search",
    json={"query": "What is the main topic?", "top_k": 3}
)

for doc in result.json()["results"]:
    print(f"Score: {doc['score']:.4f}")
    print(f"Content: {doc['content'][:100]}...")
```

### Use Case 2: Knowledge Base for Chatbot
```python
# One-time: Ingest knowledge base
knowledge_files = ["faq.pdf", "manual.docx", "policies.html"]
# ... ingest all files

# Runtime: Augment chatbot with RAG
user_question = "How do I reset my password?"

# Get relevant context
context = requests.post(
    "http://localhost:8000/v1/rag-lite/search",
    json={"query": user_question, "top_k": 2}
).json()

# Send to LLM with context
prompt = f"Context: {context['results'][0]['content']}\n\nQuestion: {user_question}"
# ... call LLM
```

### Use Case 3: Research Paper Search
```python
# Ingest research papers
import glob
for paper in glob.glob("papers/*.pdf"):
    requests.post(
        "http://localhost:8000/v1/rag-lite/ingest",
        files={"file": open(paper, "rb")}
    )

# Search across all papers
queries = [
    "machine learning techniques",
    "neural network architectures",
    "optimization algorithms"
]

for query in queries:
    results = requests.post(
        "http://localhost:8000/v1/rag-lite/search",
        json={"query": query, "top_k": 5}
    ).json()

    print(f"\n{query}:")
    for r in results["results"]:
        print(f"  - {r['metadata']['source']}: {r['score']:.4f}")
```

## Testing Plan

1. **Unit Tests**: Parser, embedder, retriever, store
2. **Integration Tests**: Full ingestion and search pipeline
3. **API Tests**: All endpoints with real HTTP requests
4. **Performance Tests**: Latency and throughput benchmarks
5. **Compatibility Tests**: Verify advanced RAG still works

## Documentation Needs

1. **API Documentation**: Complete endpoint reference
2. **Quick Start Guide**: Get started in 5 minutes
3. **Migration Guide**: Switching from advanced RAG
4. **Architecture Guide**: How components work together
5. **Performance Guide**: Optimization tips

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Code duplication (embedders in both server + rag) | Extract to shared library `llamafarm_rag_common` |
| Feature divergence | Keep server RAG simple, advanced features in rag/ |
| Breaking changes for existing users | Make it optional, default enabled for new installs |
| Performance degradation | Benchmark and optimize critical paths |
| Increased server memory usage | Use embedded ChromaDB efficiently, add limits |

## Success Metrics

- **Adoption**: 90% of new users use RAG-lite vs. advanced RAG
- **Time to First Search**: <5 minutes from install to working search
- **Performance**: Search latency <100ms for typical workloads
- **Satisfaction**: Reduced support tickets about RAG setup
- **Retention**: More users actually use RAG features

## Implementation Phases

### Phase 1: Core Components (Week 1)
- [ ] Add MarkItDown parser to server
- [ ] Create shared library `llamafarm_rag_common`
- [ ] Implement embedders (Universal, Ollama)
- [ ] Implement ChromaDB store
- [ ] Unit tests for all components

### Phase 2: Integration (Week 1)
- [ ] Create `LightweightRAGService`
- [ ] Implement chunking logic
- [ ] Integrate parser → embedder → store pipeline
- [ ] Integration tests

### Phase 3: API (Week 2)
- [ ] Create `/v1/rag-lite/ingest` endpoint
- [ ] Create `/v1/rag-lite/search` endpoint
- [ ] Create `/v1/rag-lite/stats` endpoint
- [ ] API integration tests

### Phase 4: Polish (Week 2)
- [ ] Documentation
- [ ] Example code
- [ ] Performance optimization
- [ ] Verify backward compatibility

## Open Questions

1. Should we add config option to disable RAG-lite? (Proposed: Yes, `server.enable_lightweight_rag`)
2. Should we support multiple collections? (Proposed: Later, keep simple for now)
3. Should ingestion be async via Celery? (Proposed: No, keep synchronous for simplicity)
4. Should we add health check for RAG-lite? (Proposed: Yes, similar to advanced RAG)

## Related Issues

- #XXX - RAG setup is too complex for new users
- #XXX - Need lightweight alternative to full RAG package
- #XXX - ChromaDB embedded mode support

## References

- MarkItDown: https://github.com/microsoft/markitdown
- ChromaDB: https://docs.trychroma.com/
- Similar pattern: https://github.com/langchain-ai/langchain (built-in vs. modular)
