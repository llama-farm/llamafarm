# Lightweight RAG Service

Built-in RAG system for LlamaFarm server. Provides document ingestion, embedding, and semantic search without requiring a separate worker process.

## Architecture

```
┌─────────────────────────────────────────────────┐
│ Lightweight RAG Service (server/)               │
│                                                  │
│  Parser: MarkItDown (PDF, DOCX, HTML, etc.)     │
│  Chunker: Simple text splitting (1000 chars)    │
│  Embedder: Universal Runtime HTTP API           │
│  Store: ChromaDB (embedded/persistent mode)     │
│  Retriever: Basic cosine similarity             │
└─────────────────────────────────────────────────┘
```

## Components

### Parser: MarkItDown
- **Location**: `server/services/rag/parsers/markitdown_parser.py`
- **Formats**: PDF, DOCX, PPTX, XLSX, images, HTML, text
- **OCR**: Automatic fallback for scanned PDFs/images via Universal Runtime

### Embedder: UniversalEmbedder
- **Location**: `common/llamafarm_rag_common/embedders/universal_embedder.py`
- **Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **API**: HTTP calls to Universal Runtime (http://127.0.0.1:11540)
- **Batch**: Processes up to 32 texts per request

### Vector Store: ChromaDB
- **Location**: `common/llamafarm_rag_common/stores/chroma_store.py`
- **Mode**: Embedded/persistent (no server needed)
- **Storage**: `~/.llamafarm/server_rag/chroma/`
- **Distance**: Cosine similarity

### Retriever: BasicSimilarityStrategy
- **Location**: `common/llamafarm_rag_common/retrievers/basic_similarity.py`
- **Method**: Cosine similarity search
- **Scoring**: `similarity = exp(-distance / 100)`

## API Endpoints

### `GET /v1/rag-lite/stats`
Get RAG system statistics.

**Response:**
```json
{
  "document_count": 41,
  "collection": "server_documents",
  "embedding_dimension": 384,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### `POST /v1/rag-lite/ingest`
Ingest a document into the RAG system.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload

**Response:**
```json
{
  "success": true,
  "file": "/path/to/file.pdf",
  "chunks": 34,
  "characters": 27118,
  "document_ids": ["uuid1", "uuid2", ...]
}
```

**Example (curl):**
```bash
curl -X POST http://localhost:8000/v1/rag-lite/ingest \
  -F "file=@document.pdf"
```

**Example (Python):**
```python
import requests

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post(
        "http://localhost:8000/v1/rag-lite/ingest",
        files=files
    )
    result = response.json()
    print(f"Ingested {result['chunks']} chunks")
```

### `POST /v1/rag-lite/search`
Search for documents relevant to a query.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

**Response:**
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

**Example (curl):**
```bash
curl -X POST http://localhost:8000/v1/rag-lite/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 3}'
```

**Example (Python):**
```python
import requests

payload = {"query": "What is machine learning?", "top_k": 5}
response = requests.post(
    "http://localhost:8000/v1/rag-lite/search",
    json=payload
)
result = response.json()
for doc in result['results']:
    print(f"Score: {doc['score']:.4f} - {doc['content'][:100]}...")
```

## Configuration

The service uses default configuration optimized for most use cases:

- **Project Dir**: `~/.llamafarm/server_rag/`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Universal Runtime URL**: `http://127.0.0.1:11540`
- **Chunk Size**: 1000 characters
- **Similarity Threshold**: 0.0 (no filtering)
- **Max Results**: 100

To customize, modify `server/services/rag/rag_service.py`:

```python
rag = LightweightRAGService(
    project_dir=Path("/custom/path"),
    embedding_model="different-model",
    universal_runtime_url="http://custom:port"
)
```

## Performance

Typical performance on M1 MacBook:

- **Ingestion**: ~1-2s for a 100KB PDF (parsing + chunking + embedding + storage)
- **Search**: ~50-100ms for semantic search with 40+ documents
- **Memory**: ~50MB for service + ~20MB for ChromaDB

## Comparison with Advanced RAG

| Feature | RAG-Lite (Built-in) | Advanced RAG (Optional) |
|---------|---------------------|-------------------------|
| **Availability** | Always (no worker needed) | Requires `nx start rag` |
| **Parser** | MarkItDown | Docling, LlamaIndex, etc. |
| **Vector Store** | ChromaDB (embedded) | Qdrant, Milvus, Chroma (server) |
| **Processing** | Synchronous | Celery distributed |
| **Retrieval** | Basic similarity | Reranking, hybrid, etc. |
| **Use Case** | 90% of users | 10% power users |
| **Dependencies** | 5 lightweight | Many (llama-index, etc.) |

## Testing

Run the test suite:

```bash
# Unit tests
cd server
uv run pytest tests/services/rag/ -v

# API integration tests
uv run python ../demos/test_rag_lite_api.py

# Full demo
uv run python ../demos/demo_server_rag.py

# Comparison demo
uv run python ../demos/demo_rag_comparison.py
```

## Troubleshooting

### "No module named 'markitdown'"
Install server dependencies:
```bash
cd server
uv sync
```

### "Universal Runtime not reachable"
Start Universal Runtime:
```bash
nx start universal-runtime
```

### "Failed to parse PDF"
- Check if file is readable
- For scanned PDFs, ensure Universal Runtime OCR is available
- Check server logs: `tail -f /tmp/server.log`

### "Search returns no results"
- Verify documents are ingested: `curl http://localhost:8000/v1/rag-lite/stats`
- Check embedding model is loaded in Universal Runtime
- Try lowering similarity threshold in retriever config

## Development

### Add New Parser

Create parser in `server/services/rag/parsers/`:

```python
class CustomParser:
    async def parse(self, file_path: str, metadata: dict) -> str:
        # Parse file and return text
        return parsed_text
```

Update `rag_service.py` to use new parser.

### Add New Embedder

Create embedder in `common/llamafarm_rag_common/embedders/`:

```python
class CustomEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Generate embeddings
        return embeddings
```

Update `rag_service.py` to use new embedder.

### Add New Chunker

Implement in `server/services/rag/chunkers/`:

```python
class CustomChunker:
    def chunk(self, text: str) -> list[str]:
        # Split text into chunks
        return chunks
```

## License

Part of LlamaFarm project.
