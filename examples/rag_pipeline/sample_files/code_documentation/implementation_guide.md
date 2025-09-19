# RAG System Implementation Guide

This guide walks through implementing a complete Retrieval-Augmented Generation (RAG) system from scratch.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │     Parsers      │    │   Extractors    │
│                 │───▶│                  │───▶│                 │
│ • PDF files     │    │ • PlainText      │    │ • Tables        │
│ • Markdown      │    │ • Markdown       │    │ • Links         │
│ • HTML pages    │    │ • HTML           │    │ • Entities      │
│ • CSV data      │    │ • CSV            │    │ • Summaries     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vector Store    │◀───│   Embeddings     │◀───│   Documents     │
│                 │    │                  │    │                 │
│ • ChromaDB      │    │ • Ollama         │    │ • Structured    │
│ • FAISS         │    │ • OpenAI         │    │ • Chunked       │
│ • Pinecone      │    │ • HuggingFace    │    │ • Enriched      │
│ • Qdrant        │    │ • Transformers   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Step 1: Setting Up the Project Structure

```bash
rag_system/
├── core/
│   ├── __init__.py
│   ├── base.py              # Base classes and interfaces
│   ├── pipeline.py          # Pipeline orchestration
│   └── factories.py         # Component factories
├── components/
│   ├── parsers/             # Document parsers
│   ├── extractors/          # Content extractors
│   ├── embedders/           # Embedding models
│   ├── stores/              # Vector storage
│   └── retrievers/          # Retrieval strategies
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
├── docs/
└── requirements.txt
```

## Step 2: Implementing Base Classes

### Document Class

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import uuid

@dataclass
class Document:
    """Core document representation with embeddings."""
    
    id: str
    content: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the document."""
        self.metadata[key] = value
    
    def get_content_preview(self, max_chars: int = 200) -> str:
        """Get a preview of the document content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."
```

### Parser Interface

```python
from abc import ABC, abstractmethod
from pathlib import Path

class Parser(ABC):
    """Abstract base class for all document parsers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_extensions = self._get_supported_extensions()
    
    @abstractmethod
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass
    
    @abstractmethod  
    def parse(self, file_path: str) -> List[Document]:
        """Parse a file and return documents."""
        pass
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file."""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_extensions
    
    def _create_document(self, content: str, source: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Helper to create a document with consistent metadata."""
        doc_metadata = {
            'parser': self.__class__.__name__,
            'file_size': len(content),
            'created_at': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            source=source,
            metadata=doc_metadata
        )
```

## Step 3: Implementing Specific Parsers

### Markdown Parser

```python
import re
from typing import Dict, List, Any
import frontmatter

class MarkdownParser(Parser):
    """Parser for Markdown files with structure awareness."""
    
    def _get_supported_extensions(self) -> List[str]:
        return ['.md', '.markdown']
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse markdown file with heading-based chunking."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse frontmatter if present
        post = frontmatter.loads(content)
        main_content = post.content
        metadata = post.metadata
        
        # Extract structure if enabled
        if self.config.get('chunk_by_headings', False):
            return self._chunk_by_headings(main_content, file_path, metadata)
        else:
            return self._chunk_by_size(main_content, file_path, metadata)
    
    def _chunk_by_headings(self, content: str, source: str, 
                          base_metadata: Dict[str, Any]) -> List[Document]:
        """Split content by markdown headings."""
        # Find all headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_section = ""
        current_heading = ""
        current_level = 0
        
        lines = content.split('\n')
        for line in lines:
            heading_match = re.match(heading_pattern, line)
            
            if heading_match:
                # Save previous section
                if current_section.strip():
                    sections.append({
                        'heading': current_heading,
                        'level': current_level,
                        'content': current_section.strip()
                    })
                
                # Start new section
                current_level = len(heading_match.group(1))
                current_heading = heading_match.group(2)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            sections.append({
                'heading': current_heading,
                'level': current_level,
                'content': current_section.strip()
            })
        
        # Create documents
        documents = []
        for i, section in enumerate(sections):
            section_metadata = {
                **base_metadata,
                'section_index': i,
                'heading': section['heading'],
                'heading_level': section['level']
            }
            
            doc = self._create_document(
                content=section['content'],
                source=f"{source}#{section['heading']}",
                metadata=section_metadata
            )
            documents.append(doc)
        
        return documents
```

### PDF Parser

```python
import PyPDF2
from pathlib import Path

class PDFParser(Parser):
    """Parser for PDF files with page-based chunking."""
    
    def _get_supported_extensions(self) -> List[str]:
        return ['.pdf']
    
    def parse(self, file_path: str) -> List[Document]:
        """Parse PDF file page by page."""
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    
                    if text.strip():  # Skip empty pages
                        page_metadata = {
                            'page_number': page_num + 1,
                            'total_pages': len(pdf_reader.pages),
                            'file_name': Path(file_path).name
                        }
                        
                        doc = self._create_document(
                            content=text,
                            source=f"{file_path}#page{page_num + 1}",
                            metadata=page_metadata
                        )
                        documents.append(doc)
        
        except Exception as e:
            raise ParsingError(f"Failed to parse PDF {file_path}: {str(e)}")
        
        return documents
```

## Step 4: Implementing Embedders

### Ollama Embedder

```python
import requests
import json
from typing import List

class OllamaEmbedder(Embedder):
    """Embedder using local Ollama models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model = config.get('model', 'nomic-embed-text')
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.batch_size = config.get('batch_size', 1)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama API."""
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = []
        
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                embedding = response.json()['embedding']
                embeddings.append(embedding)
            else:
                raise EmbeddingError(f"Ollama API error: {response.text}")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension by testing with sample text."""
        try:
            sample_embedding = self.embed(["test"])[0]
            return len(sample_embedding)
        except Exception as e:
            raise EmbeddingError(f"Could not determine dimension: {str(e)}")
```

### OpenAI Embedder

```python
import openai
from typing import List

class OpenAIEmbedder(Embedder):
    """Embedder using OpenAI's embedding models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model = config.get('model', 'text-embedding-ada-002')
        self.api_key = config.get('api_key')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        openai.api_key = self.api_key
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item['embedding'] for item in response['data']]
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"OpenAI API error: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Return known dimension for OpenAI models."""
        dimensions = {
            'text-embedding-ada-002': 1536,
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072
        }
        return dimensions.get(self.model, 1536)
```

## Step 5: Implementing Vector Stores

### ChromaDB Store

```python
import chromadb
from chromadb.config import Settings

class ChromaStore(VectorStore):
    """Vector store using ChromaDB."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.collection_name = config.get('collection_name', 'documents')
        self.persist_directory = config.get('persist_directory', './chroma_db')
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to ChromaDB collection."""
        try:
            ids = [doc.id for doc in documents]
            embeddings = [doc.embeddings for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            documents_content = [doc.content for doc in documents]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Search for similar documents."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            documents = []
            for i in range(len(results['ids'][0])):
                doc = Document(
                    id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                # Add search score to metadata
                doc.metadata['search_score'] = 1.0 - results['distances'][0][i]
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
```

## Step 6: Building the Pipeline

### Complete RAG Pipeline

```python
class RAGPipeline:
    """Complete RAG pipeline orchestrating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parser = None
        self.embedder = None
        self.vector_store = None
        self.extractors = []
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        # Initialize parser
        parser_config = self.config.get('parser', {})
        self.parser = create_parser(
            parser_config.get('type', 'text'),
            parser_config.get('config', {})
        )
        
        # Initialize embedder
        embedder_config = self.config.get('embedder', {})
        self.embedder = create_embedder(
            embedder_config.get('type', 'ollama'),
            embedder_config.get('name', 'default'),
            embedder_config.get('config', {})
        )
        
        # Initialize vector store
        store_config = self.config.get('vector_store', {})
        self.vector_store = create_vector_store(
            store_config.get('type', 'chroma'),
            store_config.get('name', 'default'),
            store_config.get('config', {})
        )
    
    def process_documents(self, file_paths: List[str]) -> bool:
        """Process documents through the complete pipeline."""
        all_documents = []
        
        # Parse documents
        for file_path in file_paths:
            try:
                documents = self.parser.parse(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {str(e)}")
                continue
        
        # Generate embeddings
        for doc in all_documents:
            try:
                embeddings = self.embedder.embed([doc.content])
                doc.embeddings = embeddings[0] if embeddings else []
            except Exception as e:
                logger.error(f"Failed to embed document {doc.id}: {str(e)}")
                continue
        
        # Store documents
        return self.vector_store.add_documents(all_documents)
    
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents."""
        # Generate query embedding
        query_embeddings = self.embedder.embed([query])
        if not query_embeddings:
            return []
        
        # Search vector store
        return self.vector_store.search(query_embeddings[0], top_k)
```

## Step 7: Usage Examples

### Basic Usage

```python
# Configuration
config = {
    'parser': {
        'type': 'markdown',
        'config': {
            'chunk_by_headings': True,
            'extract_frontmatter': True
        }
    },
    'embedder': {
        'type': 'ollama',
        'name': 'doc_embedder',
        'config': {
            'model': 'nomic-embed-text',
            'batch_size': 4
        }
    },
    'vector_store': {
        'type': 'chroma',
        'name': 'docs',
        'config': {
            'collection_name': 'documentation',
            'persist_directory': './vectordb'
        }
    }
}

# Initialize and run pipeline
pipeline = RAGPipeline(config)
pipeline.initialize_components()

# Process documents
file_paths = ['docs/api.md', 'docs/guide.md', 'docs/examples.md']
success = pipeline.process_documents(file_paths)

if success:
    # Search for relevant documents
    results = pipeline.search("How to implement a custom parser?", top_k=3)
    
    for doc in results:
        print(f"Score: {doc.metadata.get('search_score', 'N/A')}")
        print(f"Source: {doc.source}")
        print(f"Content: {doc.get_content_preview()}")
        print("-" * 50)
```

### Advanced Usage with Custom Components

```python
# Custom extractor
class CodeExtractor(Extractor):
    """Extract code blocks from documents."""
    
    def extract(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            code_blocks = re.findall(r'```(\w+)?\n(.*?)```', doc.content, re.DOTALL)
            doc.metadata['code_blocks'] = len(code_blocks)
            doc.metadata['languages'] = list(set(lang for lang, _ in code_blocks if lang))
        return documents

# Use in pipeline
pipeline = RAGPipeline(config)
pipeline.initialize_components()
pipeline.extractors.append(CodeExtractor({}))

# Process with extraction
success = pipeline.process_documents(file_paths)
```

This implementation guide provides a complete foundation for building a production-ready RAG system with modular, extensible components.