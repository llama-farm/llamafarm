# API Reference Documentation

## Core Classes

### Document Class

The `Document` class represents a single document in the RAG system.

```python
class Document:
    """
    Represents a document with content, metadata, and optional embeddings.
    
    Attributes:
        id (str): Unique identifier for the document
        content (str): The actual text content
        source (str): Source file or origin of the document
        metadata (Dict[str, Any]): Additional metadata
        embeddings (List[float]): Vector embeddings for semantic search
    """
    
    def __init__(self, id: str, content: str, source: str = "", 
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.source = source
        self.metadata = metadata or {}
        self.embeddings: List[float] = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "embeddings": self.embeddings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary representation."""
        doc = cls(
            id=data["id"],
            content=data["content"],
            source=data.get("source", ""),
            metadata=data.get("metadata", {})
        )
        doc.embeddings = data.get("embeddings", [])
        return doc
```

### Parser Interface

All parsers implement the `Parser` base class:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Parser(ABC):
    """Base class for all document parsers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """
        Parse a file and return a list of documents.
        
        Args:
            file_path (str): Path to the file to parse
            
        Returns:
            List[Document]: List of parsed documents
        """
        pass
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                   overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap
            
        Returns:
            List[str]: List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            
        return chunks
```

## Embedder Interface

```python
class Embedder(ABC):
    """Base class for all embedding models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            int: Embedding dimension
        """
        pass
```

## Vector Store Interface

```python
class VectorStore(ABC):
    """Base class for all vector storage backends."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents (List[Document]): Documents to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding (List[float]): Query vector
            top_k (int): Number of results to return
            
        Returns:
            List[Document]: Most similar documents
        """
        pass
    
    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            document_ids (List[str]): IDs of documents to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
```

## Factory Functions

### Parser Factory

```python
def create_parser(parser_type: str, config: Dict[str, Any]) -> Parser:
    """
    Create a parser instance based on type.
    
    Args:
        parser_type (str): Type of parser ('text', 'markdown', 'pdf', etc.)
        config (Dict[str, Any]): Parser configuration
        
    Returns:
        Parser: Parser instance
        
    Raises:
        ValueError: If parser type is not supported
    """
    parsers = {
        'text': PlainTextParser,
        'markdown': MarkdownParser,
        'pdf': PDFParser,
        'html': HTMLParser,
        'csv': CSVParser,
        'docx': DocxParser,
        'excel': ExcelParser
    }
    
    if parser_type not in parsers:
        raise ValueError(f"Unsupported parser type: {parser_type}")
    
    return parsers[parser_type](config)
```

### Embedder Factory

```python
def create_embedder(embedder_type: str, name: str, config: Dict[str, Any]) -> Embedder:
    """
    Create an embedder instance based on type.
    
    Args:
        embedder_type (str): Type of embedder ('ollama', 'openai', 'huggingface')
        name (str): Name for the embedder instance
        config (Dict[str, Any]): Embedder configuration
        
    Returns:
        Embedder: Embedder instance
        
    Raises:
        ValueError: If embedder type is not supported
    """
    embedders = {
        'ollama': OllamaEmbedder,
        'openai': OpenAIEmbedder,
        'huggingface': HuggingFaceEmbedder,
        'sentence_transformer': SentenceTransformerEmbedder
    }
    
    if embedder_type not in embedders:
        raise ValueError(f"Unsupported embedder type: {embedder_type}")
    
    return embedders[embedder_type](name, config)
```

## Configuration Examples

### Basic RAG Pipeline Setup

```python
from core.factories import create_parser, create_embedder, create_vector_store

# Initialize components
parser = create_parser('markdown', {
    'chunk_by_headings': True,
    'extract_frontmatter': True,
    'chunk_size': 1000
})

embedder = create_embedder('ollama', 'doc_embedder', {
    'model': 'nomic-embed-text',
    'batch_size': 4
})

vector_store = create_vector_store('chroma', 'docs_db', {
    'collection_name': 'documentation',
    'persist_directory': './vectordb'
})

# Process documents
documents = parser.parse('path/to/document.md')

# Generate embeddings
for doc in documents:
    embeddings = embedder.embed([doc.content])
    doc.embeddings = embeddings[0]

# Store in vector database
vector_store.add_documents(documents)
```

### Advanced Configuration

```python
# Complex parser configuration
parser_config = {
    'chunk_size': 2000,
    'overlap': 200,
    'preserve_code_blocks': True,
    'extract_links': True,
    'extract_headings': True,
    'min_chunk_size': 100
}

# Embedder with custom settings
embedder_config = {
    'model': 'all-MiniLM-L6-v2',
    'batch_size': 8,
    'normalize_embeddings': True,
    'pooling_strategy': 'mean'
}

# Vector store with advanced options
store_config = {
    'collection_name': 'advanced_docs',
    'persist_directory': './advanced_vectordb',
    'distance_metric': 'cosine',
    'index_type': 'HNSW',
    'ef_construction': 200,
    'M': 16
}
```

## Error Handling

### Common Exceptions

```python
class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass

class ParsingError(RAGError):
    """Raised when document parsing fails."""
    pass

class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass

# Usage example
try:
    documents = parser.parse('document.pdf')
except ParsingError as e:
    logger.error(f"Failed to parse document: {e}")
    # Handle parsing error
```

## Performance Optimization

### Batch Processing

```python
def process_documents_batch(file_paths: List[str], 
                          parser: Parser, 
                          embedder: Embedder,
                          vector_store: VectorStore,
                          batch_size: int = 10):
    """
    Process documents in batches for better performance.
    
    Args:
        file_paths: List of file paths to process
        parser: Parser instance
        embedder: Embedder instance
        vector_store: Vector store instance
        batch_size: Number of documents to process at once
    """
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        batch_documents = []
        
        # Parse batch
        for path in batch_paths:
            docs = parser.parse(path)
            batch_documents.extend(docs)
        
        # Generate embeddings for batch
        batch_texts = [doc.content for doc in batch_documents]
        batch_embeddings = embedder.embed(batch_texts)
        
        # Assign embeddings
        for doc, embedding in zip(batch_documents, batch_embeddings):
            doc.embeddings = embedding
        
        # Store batch
        vector_store.add_documents(batch_documents)
```

## Testing

### Unit Test Example

```python
import unittest
from unittest.mock import Mock, patch
from core.base import Document
from components.parsers.text_parser import PlainTextParser

class TestTextParser(unittest.TestCase):
    
    def setUp(self):
        self.parser = PlainTextParser(name="PlainTextParser", config={
            'chunk_size': 100,
            'overlap': 20
        })
    
    def test_parse_simple_text(self):
        """Test parsing a simple text file."""
        with patch('pathlib.Path.read_text') as mock_read:
            mock_read.return_value = "This is a test document."
            
            documents = self.parser.parse('test.txt')
            
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].content, "This is a test document.")
            self.assertEqual(documents[0].source, 'test.txt')
    
    def test_chunking(self):
        """Test text chunking functionality."""
        long_text = "A" * 250
        chunks = self.parser.chunk_text(long_text, chunk_size=100, overlap=20)
        
        self.assertEqual(len(chunks), 3)
        # Verify overlap
        self.assertTrue(chunks[0][-20:] in chunks[1][:20])

if __name__ == '__main__':
    unittest.main()
```