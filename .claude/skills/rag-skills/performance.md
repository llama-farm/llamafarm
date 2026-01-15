# RAG Performance Patterns

Best practices for batching, caching, deduplication, and optimization in RAG systems.

## Performance Overview

| Area | Impact | Key Technique |
|------|--------|---------------|
| Embedding Generation | High | Batch processing |
| Vector Storage | Medium | Deduplication |
| Search | Medium | Result caching |
| Document Parsing | Low | Parallel processing |

## Batch Processing

### Embedding Batching

Location: `rag/core/base.py`, `rag/core/ingest_handler.py`

```python
class Embedder(Component):
    def process(self, documents: list[Document]) -> ProcessingResult:
        """Batch embed documents for efficiency."""
        # Extract all texts at once
        texts = [doc.content for doc in documents]

        # Single batch call to embedding API
        embeddings = self.embed(texts)

        # Assign embeddings back
        for doc, embedding in zip(documents, embeddings, strict=False):
            doc.embeddings = embedding

        return ProcessingResult(
            documents=documents,
            metrics={"embedded_count": len(documents)},
        )
```

### Optimal Batch Sizes

| Embedder Type | Recommended Batch | Max Batch |
|---------------|-------------------|-----------|
| Ollama | 32 | 64 |
| OpenAI | 100 | 2048 |
| Universal (local) | 16 | 32 |
| HuggingFace | 32 | 64 |

## Deduplication

### Hash-Based Deduplication

Location: `rag/utils/hash_utils.py`, `rag/components/stores/chroma_store/`

```python
class DeduplicationTracker:
    """Track document and chunk hashes to prevent duplicates."""

    def __init__(self):
        self._document_hashes: set[str] = set()
        self._chunk_hashes: set[str] = set()
        self._source_hashes: set[str] = set()

    def is_duplicate_document(self, doc_hash: str) -> bool:
        return doc_hash in self._document_hashes

    def is_duplicate_chunk(self, chunk_hash: str) -> bool:
        return chunk_hash in self._chunk_hashes

    def register_document(self, doc_hash: str, doc_id: str, source_hash: str):
        self._document_hashes.add(doc_hash)
        if source_hash:
            self._source_hashes.add(source_hash)

    def register_chunk(self, chunk_hash: str, doc_id: str):
        self._chunk_hashes.add(chunk_hash)
```

### Document ID Generation

```python
import hashlib

def generate_document_id(file_hash: str, chunk_index: int) -> str:
    """Generate deterministic ID from file hash and chunk index."""
    return f"{file_hash[:16]}_{chunk_index:04d}"

# In ingest_handler.py
file_hash = hashlib.sha256(file_data).hexdigest()
for i, doc in enumerate(documents):
    doc.id = generate_document_id(file_hash, i)
    doc.metadata["file_hash"] = file_hash
    doc.metadata["chunk_index"] = i
```

## Circuit Breaker Pattern

### Embedder Safety

Location: `rag/utils/embedding_safety.py`

```python
class CircuitBreaker:
    """Prevent cascading failures from unavailable embedders."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open allows one attempt

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self, error: Exception | None = None):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
```

### Embedding Validation

```python
def is_valid_embedding(
    embedding: list[float],
    expected_dimension: int | None = None,
    allow_zero: bool = False,
) -> tuple[bool, str | None]:
    """Validate embedding vector."""
    if not embedding:
        return False, "Empty embedding"

    if expected_dimension and len(embedding) != expected_dimension:
        return False, f"Wrong dimension: {len(embedding)} != {expected_dimension}"

    if not allow_zero and is_zero_vector(embedding):
        return False, "Zero vector"

    return True, None

def is_zero_vector(embedding: list[float], threshold: float = 1e-10) -> bool:
    """Check if embedding is effectively zero."""
    return all(abs(v) < threshold for v in embedding)
```

## Code Review Checklist

### 1. Batch Embedding Processing

**Description**: Embeddings should be generated in batches, not one at a time.

**Search Pattern**:
```bash
grep -rn "for.*doc.*in.*documents.*embed\|embed\(\[doc" rag/
```

**Pass Criteria**:
- Texts extracted as list
- Single embed() call for batch
- Batch size is configurable

**Fail Criteria**:
- embed() called per document
- No batching logic
- Hardcoded batch size

**Severity**: High

**Recommendation**: Batch texts before calling embed().

---

### 2. Deduplication Enabled

**Description**: Duplicate detection should prevent redundant storage.

**Search Pattern**:
```bash
grep -rn "enable_deduplication\|DeduplicationTracker\|file_hash" rag/
```

**Pass Criteria**:
- Deduplication tracker initialized
- Hash checked before storage
- Configurable via settings

**Fail Criteria**:
- No deduplication
- Check after storage
- Always disabled

**Severity**: Medium

**Recommendation**: Enable deduplication by default with content hashing.

---

### 3. Circuit Breaker Configuration

**Description**: Embedders should have circuit breaker protection.

**Search Pattern**:
```bash
grep -rn "CircuitBreaker\|failure_threshold\|circuit_breaker" rag/
```

**Pass Criteria**:
- CircuitBreaker in embedder base
- Configurable threshold
- Proper state transitions

**Fail Criteria**:
- No circuit breaker
- Hardcoded thresholds
- Missing state management

**Severity**: High

**Recommendation**: Configure circuit breaker with 5 failures, 60s reset.

---

### 4. Embedding Validation

**Description**: Embeddings must be validated before storage.

**Search Pattern**:
```bash
grep -rn "is_valid_embedding\|is_zero_vector\|embedding.*validation" rag/
```

**Pass Criteria**:
- Validation before storage
- Zero vector rejection
- Dimension check

**Fail Criteria**:
- No validation
- Zero vectors accepted
- Wrong dimensions stored

**Severity**: High

**Recommendation**: Validate all embeddings with is_valid_embedding().

---

### 5. Deterministic Document IDs

**Description**: Document IDs should be deterministic for deduplication.

**Search Pattern**:
```bash
grep -rn "doc.id.*=.*hash\|file_hash.*chunk" rag/
```

**Pass Criteria**:
- ID derived from content hash
- Chunk index included
- Consistent generation

**Fail Criteria**:
- Random UUIDs always
- Missing chunk index
- Inconsistent format

**Severity**: Medium

**Recommendation**: Use `{file_hash[:16]}_{chunk_index:04d}` format.

---

### 6. Fail-Fast Configuration

**Description**: Embedder failures should stop processing early.

**Search Pattern**:
```bash
grep -rn "fail_fast\|EmbedderUnavailableError" rag/
```

**Pass Criteria**:
- fail_fast configurable
- Exception raised on failure
- Partial results returned

**Fail Criteria**:
- Silent failures
- Zero vectors substituted
- No error propagation

**Severity**: High

**Recommendation**: Enable fail_fast=True for production.

---

### 7. Search Result Limiting

**Description**: Search results should be properly limited.

**Search Pattern**:
```bash
grep -rn "top_k\|max_results\|n_results" rag/
```

**Pass Criteria**:
- top_k parameter respected
- Max limit configured
- Pagination supported

**Fail Criteria**:
- Unlimited results
- top_k ignored
- No pagination

**Severity**: Medium

**Recommendation**: Enforce max_results limit (e.g., 100).

## Anti-Patterns

### Per-Document Embedding

```python
# BAD: N API calls for N documents
for doc in documents:
    embedding = embedder.embed([doc.content])[0]
    doc.embeddings = embedding

# GOOD: Single batch API call
texts = [doc.content for doc in documents]
embeddings = embedder.embed(texts)
for doc, emb in zip(documents, embeddings, strict=False):
    doc.embeddings = emb
```

### Random Document IDs

```python
# BAD: Always new UUID - no deduplication possible
doc.id = str(uuid.uuid4())

# GOOD: Deterministic ID from content
file_hash = hashlib.sha256(file_data).hexdigest()
doc.id = f"{file_hash[:16]}_{chunk_index:04d}"
```

### Ignoring Circuit Breaker

```python
# BAD: Keep trying despite failures
def embed(self, texts):
    for text in texts:
        try:
            return self._call_api(text)
        except:
            continue  # Never stops!

# GOOD: Circuit breaker stops cascade
def embed_text(self, text):
    self.check_circuit_breaker()  # May raise
    try:
        result = self._call_api(text)
        self.record_success()
        return result
    except Exception as e:
        self.record_failure(e)
        raise
```

### Storing Invalid Embeddings

```python
# BAD: Store whatever comes back
doc.embeddings = embedder.embed([doc.content])[0]
store.add_documents([doc])

# GOOD: Validate before storage
embedding = embedder.embed([doc.content])[0]
is_valid, error = is_valid_embedding(embedding, expected_dim)
if not is_valid:
    raise ValueError(f"Invalid embedding: {error}")
doc.embeddings = embedding
store.add_documents([doc])
```

## Performance Metrics

Track these metrics for RAG performance:

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Embedding latency (batch) | <1s per 32 docs | >5s |
| Storage latency | <500ms per batch | >2s |
| Search latency | <100ms | >500ms |
| Deduplication hit rate | Varies | N/A |
| Circuit breaker trips | 0 | >1/hour |

## Configuration Recommendations

```python
# Production settings
EMBEDDER_CONFIG = {
    "batch_size": 32,
    "fail_fast": True,
    "circuit_breaker": {
        "failure_threshold": 5,
        "reset_timeout": 60.0,
    },
}

VECTOR_STORE_CONFIG = {
    "enable_deduplication": True,
    "distance_metric": "cosine",
    "max_results": 100,
}

RETRIEVAL_CONFIG = {
    "similarity_threshold": 0.5,
    "max_results": 100,
}
```
