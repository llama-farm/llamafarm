# ChromaDB Patterns for RAG

Best practices for vector storage, collection management, and similarity search using ChromaDB.

## Architecture Overview

```
Documents with Embeddings -> ChromaStore -> Collection -> HNSW Index
                                              |
                                              v
                                    Persistent Storage
```

## Key Classes

### ChromaStore Implementation

Location: `rag/components/stores/chroma_store/chroma_store.py`

```python
class ChromaStore(VectorStore):
    # Class-level client cache for thread safety
    _client_cache: dict[str, chromadb.ClientAPI] = {}
    _client_cache_lock = threading.Lock()
    _collection_setup_lock = threading.Lock()

    def __init__(
        self,
        name: str = "ChromaStore",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        self.collection_name = config.get("collection_name", "documents")
        self.host = config.get("host") or os.getenv("CHROMADB_HOST")
        self.port = config.get("port") or os.getenv("CHROMADB_PORT")
        self.distance_metric = config.get("distance_metric", "cosine")

        # Initialize client with caching
        if self.host and self.port:
            self.client = self._get_or_create_client(
                f"http://{self.host}:{self.port}",
                lambda: chromadb.HttpClient(host=self.host, port=self.port)
            )
        else:
            self.client = self._get_or_create_client(
                f"persistent://{self.persist_directory}",
                lambda: chromadb.PersistentClient(path=self.persist_directory)
            )

        self._setup_collection()
```

### Thread-Safe Client Caching

```python
@classmethod
def _get_or_create_client(
    cls, client_key: str, client_factory
) -> chromadb.ClientAPI:
    with cls._client_cache_lock:
        if client_key not in cls._client_cache:
            logger.info(f"Creating new ChromaDB client for: {client_key}")
            cls._client_cache[client_key] = client_factory()
        return cls._client_cache[client_key]
```

## Distance Metrics

| Metric | Use Case | Score Interpretation |
|--------|----------|---------------------|
| `cosine` | Text embeddings (default) | 0=identical, 2=opposite |
| `l2` | Euclidean distance | 0=identical, larger=different |
| `ip` | Inner product | Higher=more similar |

## Metadata Constraints

ChromaDB only accepts these metadata types:
- `str`
- `int`
- `float`
- `bool`

**NOT supported**: `list`, `dict`, `None`

### Metadata Cleaning Pattern

```python
def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue  # Skip None
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            # Convert to comma-separated string
            cleaned[key] = ",".join(str(v) for v in value if v is not None)
        elif isinstance(value, dict):
            # Convert to JSON string
            cleaned[key] = json.dumps(value)
        else:
            cleaned[key] = str(value)
    return cleaned
```

## Code Review Checklist

### 1. Thread-Safe Client Access

**Description**: ChromaDB clients must be shared safely across threads.

**Search Pattern**:
```bash
grep -rn "_client_cache\|_client_cache_lock\|threading.Lock" rag/components/stores/
```

**Pass Criteria**:
- Client cache uses class-level storage
- Lock protects cache access
- Single client per database path

**Fail Criteria**:
- New client created per request
- No locking on cache
- Multiple clients for same path

**Severity**: Critical

**Recommendation**: Use class-level client cache with threading.Lock.

---

### 2. Collection Setup Atomicity

**Description**: Collection creation should be atomic to prevent race conditions.

**Search Pattern**:
```bash
grep -rn "get_or_create_collection\|_collection_setup_lock" rag/components/stores/
```

**Pass Criteria**:
- Uses get_or_create_collection()
- Lock protects setup
- Metadata set during creation

**Fail Criteria**:
- Separate get + create calls
- No locking on setup
- Metadata set after creation

**Severity**: High

**Recommendation**: Use get_or_create_collection() within a lock.

---

### 3. Distance Metric Configuration

**Description**: Distance metric should be configurable and validated.

**Search Pattern**:
```bash
grep -rn "distance_metric\|hnsw:space" rag/components/stores/
```

**Pass Criteria**:
- Metric from config with default
- Validation against valid metrics
- Correct HNSW space mapping

**Fail Criteria**:
- Hardcoded metric
- No validation
- Wrong metric name mapping

**Severity**: Medium

**Recommendation**: Validate metric is one of: cosine, l2, ip.

---

### 4. Metadata Type Safety

**Description**: Metadata values must be ChromaDB-compatible types.

**Search Pattern**:
```bash
grep -rn "cleaned_metadata\|isinstance.*str.*int.*float.*bool" rag/components/stores/
```

**Pass Criteria**:
- All values converted to supported types
- None values skipped
- Lists/dicts serialized to strings

**Fail Criteria**:
- Raw metadata passed to ChromaDB
- None values included
- Complex types not converted

**Severity**: High

**Recommendation**: Clean all metadata before add_documents().

---

### 5. Similarity Score Conversion

**Description**: Distance scores must be converted to similarity scores correctly.

**Search Pattern**:
```bash
grep -rn "similarity_score\|distance.*score" rag/components/stores/
```

**Pass Criteria**:
- Distance-to-similarity conversion per metric
- Score added to document metadata
- Consistent score range (0-1)

**Fail Criteria**:
- Raw distances returned as scores
- Missing metric-specific conversion
- Inconsistent score interpretation

**Severity**: Medium

**Recommendation**: Use metric-appropriate conversion formulas.

---

### 6. Deduplication Support

**Description**: Duplicate detection should prevent redundant storage.

**Search Pattern**:
```bash
grep -rn "DeduplicationTracker\|document_hash\|chunk_hash\|_document_exists" rag/
```

**Pass Criteria**:
- Hash-based duplicate detection
- Check before add
- Configurable enable/disable

**Fail Criteria**:
- No deduplication
- Check after add
- Always enabled without config

**Severity**: Medium

**Recommendation**: Use content hash for deduplication with configurable toggle.

---

### 7. Error Handling in Search

**Description**: Search errors should be handled gracefully.

**Search Pattern**:
```bash
grep -rn "def search.*:$" -A 30 rag/components/stores/
```

**Pass Criteria**:
- Try/except around query
- Empty list on error
- Error logged with context

**Fail Criteria**:
- Exceptions propagate
- None returned on error
- Silent failures

**Severity**: High

**Recommendation**: Return empty list on search errors with logging.

## Anti-Patterns

### Creating Client Per Request

```python
# BAD: New client every time
def search(self, query_embedding):
    client = chromadb.PersistentClient(path=self.persist_directory)
    collection = client.get_collection(self.collection_name)
    return collection.query(...)

# GOOD: Cached client
def search(self, query_embedding):
    return self.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
```

### Passing Raw Metadata

```python
# BAD: Raw metadata with unsupported types
self.collection.add(
    ids=[doc.id],
    embeddings=[doc.embeddings],
    metadatas=[doc.metadata],  # May contain lists, dicts, None!
)

# GOOD: Cleaned metadata
cleaned = self._clean_metadata(doc.metadata)
self.collection.add(
    ids=[doc.id],
    embeddings=[doc.embeddings],
    metadatas=[cleaned],
)
```

### Ignoring Distance Metric

```python
# BAD: Hardcoded score conversion
similarity = 1 - distance

# GOOD: Metric-aware conversion
if self.distance_metric == "cosine":
    similarity = 1.0 - (distance / 2.0)
elif self.distance_metric == "l2":
    similarity = 1.0 / (1.0 + distance / 100.0)
elif self.distance_metric == "ip":
    similarity = (1.0 + distance) / 2.0
```

## Production Recommendations

1. **Use HTTP Client in Production**: PersistentClient has SQLite locking issues
2. **Set Appropriate Collection Size**: Monitor collection count
3. **Regular Backups**: ChromaDB persist_directory should be backed up
4. **Monitor Embedding Dimensions**: Ensure consistent dimensions across adds
