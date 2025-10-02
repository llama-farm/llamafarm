# RAG System Best Practices and Design Patterns

## Architecture Principles

### 1. Separation of Concerns

```python
# ❌ Bad: Monolithic class handling everything
class RAGSystem:
    def parse_and_embed_and_store(self, file_path):
        # Parse document
        content = self.parse_file(file_path)
        # Generate embeddings
        embeddings = self.create_embeddings(content)
        # Store in database
        self.store_vectors(embeddings)

# ✅ Good: Separate responsibilities
class DocumentProcessor:
    def __init__(self, parser: Parser, embedder: Embedder, store: VectorStore):
        self.parser = parser
        self.embedder = embedder  
        self.store = store
    
    def process(self, file_path: str) -> bool:
        documents = self.parser.parse(file_path)
        for doc in documents:
            doc.embeddings = self.embedder.embed([doc.content])[0]
        return self.store.add_documents(documents)
```

### 2. Interface Segregation

```python
# ✅ Small, focused interfaces
class Searchable(Protocol):
    def search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        ...

class Storable(Protocol):
    def add_documents(self, documents: List[Document]) -> bool:
        ...

class Retrievable(Protocol):
    def get_document(self, doc_id: str) -> Optional[Document]:
        ...

# ✅ Implementations can choose which interfaces to support
class ChromaStore(Searchable, Storable, Retrievable):
    # Implements all three interfaces
    pass

class ReadOnlyStore(Searchable, Retrievable):
    # Only implements search and retrieval
    pass
```

### 3. Dependency Injection

```python
# ✅ Inject dependencies rather than creating them
class RAGPipeline:
    def __init__(self, 
                 parser: Parser,
                 embedder: Embedder, 
                 vector_store: VectorStore,
                 extractors: List[Extractor] = None):
        self.parser = parser
        self.embedder = embedder
        self.vector_store = vector_store
        self.extractors = extractors or []

# ✅ Use factories for complex initialization
def create_rag_pipeline(config: Dict[str, Any]) -> RAGPipeline:
    parser = ParserFactory.create(config['parser'])
    embedder = EmbedderFactory.create(config['embedder'])
    store = VectorStoreFactory.create(config['vector_store'])
    extractors = [ExtractorFactory.create(ext) for ext in config.get('extractors', [])]
    
    return RAGPipeline(parser, embedder, store, extractors)
```

## Performance Optimization

### 1. Batch Processing

```python
class BatchProcessor:
    """Process documents in batches for better performance."""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
    
    def process_documents(self, documents: List[Document], 
                         embedder: Embedder) -> List[Document]:
        # ✅ Process embeddings in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_texts = [doc.content for doc in batch]
            
            # Single API call for the batch
            batch_embeddings = embedder.embed(batch_texts)
            
            # Assign embeddings back to documents
            for doc, embedding in zip(batch, batch_embeddings):
                doc.embeddings = embedding
        
        return documents
```

### 2. Lazy Loading and Caching

```python
class CachedEmbedder:
    """Embedder with caching to avoid recomputing embeddings."""
    
    def __init__(self, embedder: Embedder, cache_size: int = 1000):
        self.embedder = embedder
        self.cache = {}
        self.cache_size = cache_size
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if text_hash in self.cache:
                results.append(self.cache[text_hash])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # Placeholder
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.embedder.embed(uncached_texts)
            
            # Update cache and results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                text_hash = hash(texts[idx])
                self.cache[text_hash] = embedding
                results[idx] = embedding
        
        # Cleanup cache if too large
        if len(self.cache) > self.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.cache_size]
            for key in keys_to_remove:
                del self.cache[key]
        
        return results
```

### 3. Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGPipeline:
    """Asynchronous RAG pipeline for better concurrency."""
    
    def __init__(self, parser: Parser, embedder: Embedder, store: VectorStore):
        self.parser = parser
        self.embedder = embedder
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_files(self, file_paths: List[str]) -> List[Document]:
        """Process multiple files concurrently."""
        # Parse files concurrently
        parse_tasks = [
            asyncio.get_event_loop().run_in_executor(
                self.executor, self.parser.parse, file_path
            ) for file_path in file_paths
        ]
        
        parsed_results = await asyncio.gather(*parse_tasks)
        all_documents = [doc for docs in parsed_results for doc in docs]
        
        # Generate embeddings in batches
        await self._generate_embeddings_async(all_documents)
        
        return all_documents
    
    async def _generate_embeddings_async(self, documents: List[Document]):
        """Generate embeddings asynchronously."""
        batch_size = 10
        
        embedding_tasks = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            task = asyncio.get_event_loop().run_in_executor(
                self.executor, self._embed_batch, batch
            )
            embedding_tasks.append(task)
        
        await asyncio.gather(*embedding_tasks)
    
    def _embed_batch(self, documents: List[Document]):
        """Synchronous batch embedding for thread executor."""
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embeddings = embedding
```

## Error Handling and Resilience

### 1. Graceful Degradation

```python
class ResilientEmbedder:
    """Embedder with fallback strategies."""
    
    def __init__(self, primary: Embedder, fallback: Embedder):
        self.primary = primary
        self.fallback = fallback
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.primary.embed(texts)
        except Exception as e:
            logger.warning(f"Primary embedder failed: {e}. Using fallback.")
            try:
                return self.fallback.embed(texts)
            except Exception as fallback_error:
                logger.error(f"Fallback embedder also failed: {fallback_error}")
                # Return zero vectors as last resort
                dimension = self.primary.get_embedding_dimension()
                return [[0.0] * dimension for _ in texts]
```

### 2. Circuit Breaker Pattern

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Usage
class RobustEmbedder:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.circuit_breaker = CircuitBreaker()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.circuit_breaker.call(self.embedder.embed, texts)
```

### 3. Retry Logic with Exponential Backoff

```python
import time
import random
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic with exponential backoff."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

# Usage
class ReliableEmbedder:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed(texts)
```

## Security Best Practices

### 1. Input Validation and Sanitization

```python
import re
from pathlib import Path

class SecureParser:
    """Parser with security validations."""
    
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    def __init__(self, parser: Parser):
        self.parser = parser
    
    def parse(self, file_path: str) -> List[Document]:
        # Validate file path
        self._validate_file_path(file_path)
        
        # Validate file size
        self._validate_file_size(file_path)
        
        # Parse with original parser
        documents = self.parser.parse(file_path)
        
        # Sanitize content
        for doc in documents:
            doc.content = self._sanitize_content(doc.content)
        
        return documents
    
    def _validate_file_path(self, file_path: str):
        """Validate file path for security."""
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed: {path.suffix}")
        
        # Prevent path traversal
        if '..' in str(path) or str(path).startswith('/'):
            raise ValueError("Invalid file path")
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
    
    def _validate_file_size(self, file_path: str):
        """Check file size limits."""
        size = Path(file_path).stat().st_size
        if size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {size} bytes")
    
    def _sanitize_content(self, content: str) -> str:
        """Remove potentially harmful content."""
        # Remove potential script tags
        content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove other potentially dangerous HTML
        content = re.sub(r'<(iframe|object|embed).*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        return content
```

### 2. API Key and Credential Management

```python
import os
from typing import Optional

class CredentialManager:
    """Secure credential management."""
    
    @staticmethod
    def get_api_key(service: str) -> Optional[str]:
        """Get API key from environment or secure store."""
        # Try environment variable first
        env_var = f"{service.upper()}_API_KEY"
        api_key = os.getenv(env_var)
        
        if not api_key:
            # Try alternative naming
            api_key = os.getenv(f"{service}_KEY")
        
        if not api_key:
            logger.warning(f"No API key found for {service}")
        
        return api_key
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Basic API key validation."""
        if not api_key:
            return False
        
        # Check minimum length
        if len(api_key) < 10:
            return False
        
        # Check for obvious test keys
        test_patterns = ['test', 'demo', 'example', 'fake']
        if any(pattern in api_key.lower() for pattern in test_patterns):
            return False
        
        return True

# Usage
class SecureOpenAIEmbedder(OpenAIEmbedder):
    def __init__(self, name: str, config: Dict[str, Any]):
        api_key = CredentialManager.get_api_key('openai')
        
        if not CredentialManager.validate_api_key(api_key):
            raise ValueError("Invalid or missing OpenAI API key")
        
        config['api_key'] = api_key
        super().__init__(name, config)
```

## Testing Strategies

### 1. Unit Testing with Mocks

```python
import unittest
from unittest.mock import Mock, patch
import pytest

class TestDocumentProcessor(unittest.TestCase):
    
    def setUp(self):
        self.mock_parser = Mock(spec=Parser)
        self.mock_embedder = Mock(spec=Embedder)
        self.mock_store = Mock(spec=VectorStore)
        
        self.processor = DocumentProcessor(
            self.mock_parser,
            self.mock_embedder,
            self.mock_store
        )
    
    def test_successful_processing(self):
        # Arrange
        test_doc = Document(id="test-1", content="Test content")
        self.mock_parser.parse.return_value = [test_doc]
        self.mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
        self.mock_store.add_documents.return_value = True
        
        # Act
        result = self.processor.process("test.txt")
        
        # Assert
        self.assertTrue(result)
        self.mock_parser.parse.assert_called_once_with("test.txt")
        self.mock_embedder.embed.assert_called_once_with(["Test content"])
        self.mock_store.add_documents.assert_called_once()
        
        # Verify embedding was assigned
        stored_docs = self.mock_store.add_documents.call_args[0][0]
        self.assertEqual(stored_docs[0].embeddings, [0.1, 0.2, 0.3])
```

### 2. Integration Testing

```python
@pytest.mark.integration
class TestRAGPipelineIntegration:
    
    @pytest.fixture
    def pipeline(self):
        config = {
            'parser': {'type': 'text', 'config': {}},
            'embedder': {'type': 'mock', 'config': {}},
            'vector_store': {'type': 'memory', 'config': {}}
        }
        return RAGPipeline(config)
    
    def test_end_to_end_processing(self, pipeline, tmp_path):
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        # Process document
        success = pipeline.process_documents([str(test_file)])
        assert success
        
        # Search for content
        results = pipeline.search("test document", top_k=1)
        assert len(results) == 1
        assert "test document" in results[0].content.lower()
```

### 3. Performance Testing

```python
import time
import psutil
from memory_profiler import profile

class PerformanceMonitor:
    """Monitor performance metrics during testing."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
    
    def stop(self) -> Dict[str, float]:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        return {
            'elapsed_time': end_time - self.start_time,
            'memory_delta': (end_memory - self.start_memory) / 1024 / 1024,  # MB
            'peak_memory': psutil.Process().memory_info().rss / 1024 / 1024   # MB
        }

def test_large_document_processing():
    """Test performance with large documents."""
    pipeline = RAGPipeline(test_config)
    monitor = PerformanceMonitor()
    
    # Generate large test documents
    large_documents = [
        Document(id=f"doc-{i}", content="Lorem ipsum " * 1000)
        for i in range(100)
    ]
    
    monitor.start()
    
    # Process documents
    for doc in large_documents:
        doc.embeddings = pipeline.embedder.embed([doc.content])[0]
    
    pipeline.vector_store.add_documents(large_documents)
    
    metrics = monitor.stop()
    
    # Assert performance criteria
    assert metrics['elapsed_time'] < 60.0  # Should complete in under 1 minute
    assert metrics['memory_delta'] < 500   # Should not use more than 500MB additional memory
```

This comprehensive guide covers essential best practices for building robust, secure, and performant RAG systems.