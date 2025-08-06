# RAG System Metadata Best Practices

## Overview

Proper metadata management is crucial for effective RAG (Retrieval-Augmented Generation) systems. This document outlines best practices for metadata handling across all components and strategies.

## Core Metadata Fields

Every document in the RAG system should include these essential metadata fields:

### 1. **Document Identification**
```python
{
    "id": "unique_document_id",           # Unique identifier (hash-based or UUID)
    "source": "path/to/document.pdf",     # Original file path or URL
    "source_type": "pdf",                 # File type (pdf, html, csv, etc.)
    "content_hash": "sha256_hash",        # Content hash for deduplication
}
```

### 2. **Temporal Information**
```python
{
    "created_date": "2024-01-15",         # Document creation date
    "modified_date": "2024-02-20",        # Last modification date
    "processing_date": "2024-03-01",      # When processed by RAG
    "processing_timestamp": "2024-03-01T10:30:00Z",  # ISO format timestamp
}
```

### 3. **Content Description**
```python
{
    "title": "Document Title",            # Document title
    "summary": "Brief document summary",  # Auto-generated or provided
    "language": "en",                     # Language code
    "word_count": 1500,                   # Number of words
    "page_count": 10,                     # Number of pages (if applicable)
}
```

### 4. **Classification & Organization**
```python
{
    "category": "technical",              # Document category
    "tags": ["ai", "machine-learning"],   # Relevant tags
    "department": "engineering",          # Organizational unit
    "access_level": "public",             # Access control
}
```

### 5. **Processing Information**
```python
{
    "parser_type": "PDFParser",           # Parser used
    "extractors": ["entities", "summary"], # Extractors applied
    "chunk_size": 1000,                   # Chunking parameters
    "chunk_overlap": 200,                 # Overlap size
    "embedding_model": "nomic-embed-text", # Embedding model used
}
```

## Strategy-Specific Metadata

Different use cases require additional metadata:

### Research Papers
```python
{
    "authors": ["Smith, J.", "Doe, A."],
    "publication_date": "2024-01-15",
    "journal": "Nature AI",
    "doi": "10.1038/nature.2024.12345",
    "citations": 45,
    "abstract": "Paper abstract...",
    "keywords": ["transformers", "scaling"],
}
```

### Customer Support
```python
{
    "ticket_id": "SUPPORT-12345",
    "customer_id": "CUST-789",
    "priority": "high",
    "status": "resolved",
    "resolution_time": "2h 30m",
    "agent_id": "AGENT-456",
    "satisfaction_score": 4.5,
}
```

### Legal Documents
```python
{
    "case_number": "2024-CV-12345",
    "jurisdiction": "US-CA",
    "document_type": "contract",
    "parties": ["Company A", "Company B"],
    "effective_date": "2024-01-01",
    "expiration_date": "2025-01-01",
    "confidentiality": "confidential",
}
```

### Code Documentation
```python
{
    "version": "2.1.0",
    "api_version": "v2",
    "deprecated": false,
    "last_reviewed": "2024-02-15",
    "code_examples": 5,
    "related_apis": ["auth", "users"],
}
```

## Implementation Guidelines

### 1. **Metadata Extraction Pipeline**

```python
from typing import Dict, Any, List
from datetime import datetime
import hashlib

class MetadataEnricher:
    """Enrich documents with comprehensive metadata."""
    
    def enrich_document(self, doc: Document) -> Document:
        """Add essential metadata to document."""
        
        # Generate document ID if not present
        if not doc.id:
            doc.id = self._generate_document_id(doc)
        
        # Add content hash
        doc.metadata["content_hash"] = self._generate_content_hash(doc.content)
        
        # Add temporal metadata
        doc.metadata["processing_timestamp"] = datetime.now().isoformat()
        doc.metadata["processing_date"] = datetime.now().strftime("%Y-%m-%d")
        
        # Add content statistics
        doc.metadata["word_count"] = len(doc.content.split())
        doc.metadata["char_count"] = len(doc.content)
        
        # Add source information
        if doc.source:
            doc.metadata["source_type"] = self._detect_source_type(doc.source)
            doc.metadata["file_name"] = Path(doc.source).name
        
        return doc
    
    def _generate_document_id(self, doc: Document) -> str:
        """Generate unique document ID."""
        content = f"{doc.source or 'unknown'}:{doc.content[:100]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
```

### 2. **Metadata Validation**

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

class DocumentMetadata(BaseModel):
    """Validate document metadata."""
    
    # Required fields
    id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Document source")
    processing_timestamp: str = Field(..., description="Processing timestamp")
    
    # Optional but recommended
    title: Optional[str] = None
    summary: Optional[str] = None
    content_hash: Optional[str] = None
    word_count: Optional[int] = None
    
    # Domain-specific
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    
    # Extractor results
    extractors: Optional[Dict[str, Any]] = {}
    
    @validator('processing_timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is in ISO format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except:
            raise ValueError("Timestamp must be in ISO format")
        return v
```

### 3. **Metadata Storage Considerations**

**ChromaDB Compatibility:**
- ChromaDB only accepts primitive types: `str`, `int`, `float`, `bool`, `None`
- Complex types must be serialized:

```python
def prepare_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to ChromaDB-compatible format."""
    chroma_metadata = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            chroma_metadata[key] = value
        elif isinstance(value, (list, tuple)):
            # Convert lists to JSON strings
            chroma_metadata[f"{key}_json"] = json.dumps(value)
            chroma_metadata[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            # Flatten nested dictionaries
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)):
                    chroma_metadata[f"{key}_{sub_key}"] = sub_value
        elif isinstance(value, datetime):
            chroma_metadata[key] = value.isoformat()
        else:
            # Convert other types to string
            chroma_metadata[key] = str(value)
    
    return chroma_metadata
```

### 4. **Metadata in Search & Retrieval**

```python
class MetadataAwareRetriever:
    """Retriever that leverages metadata for better results."""
    
    def search_with_metadata_filters(
        self,
        query: str,
        metadata_filters: Dict[str, Any],
        top_k: int = 10
    ) -> List[Document]:
        """Search with metadata filtering."""
        
        # Example filters
        filters = {
            "$and": [
                {"category": {"$eq": metadata_filters.get("category")}},
                {"processing_date": {"$gte": metadata_filters.get("date_from")}},
                {"word_count": {"$lte": metadata_filters.get("max_words", 10000)}}
            ]
        }
        
        return self.vector_store.search(
            query_embedding=self.embed(query),
            filter=filters,
            top_k=top_k
        )
```

## Best Practices Summary

1. **Always Include Core Metadata**
   - Document ID, source, processing timestamp
   - Content hash for deduplication
   - Basic content statistics

2. **Use Consistent Naming**
   - Use snake_case for field names
   - Be descriptive but concise
   - Follow domain conventions

3. **Validate Metadata**
   - Use Pydantic models for validation
   - Ensure required fields are present
   - Validate data types and formats

4. **Handle Complex Types**
   - Serialize complex types for storage
   - Maintain both human-readable and searchable formats
   - Consider storage backend limitations

5. **Enable Metadata Filtering**
   - Design metadata for common query patterns
   - Index frequently filtered fields
   - Balance granularity with performance

6. **Version Your Metadata Schema**
   - Track metadata schema changes
   - Provide migration paths
   - Document breaking changes

## Example: Complete Document with Metadata

```python
document = Document(
    id="doc_9a8b7c6d5e4f",
    content="This is the main content of the document...",
    source="research_papers/transformers_2024.pdf",
    metadata={
        # Core identification
        "id": "doc_9a8b7c6d5e4f",
        "source": "research_papers/transformers_2024.pdf",
        "source_type": "pdf",
        "content_hash": "sha256_abcdef123456",
        
        # Temporal
        "created_date": "2024-01-15",
        "processing_timestamp": "2024-03-01T10:30:00Z",
        "processing_date": "2024-03-01",
        
        # Content description
        "title": "Advances in Transformer Architecture",
        "summary": "This paper explores recent improvements...",
        "language": "en",
        "word_count": 5420,
        "page_count": 12,
        
        # Classification
        "category": "research",
        "tags": ["transformers", "attention", "nlp"],
        "department": "ai_research",
        
        # Domain-specific (research paper)
        "authors": ["Smith, J.", "Doe, A.", "Johnson, B."],
        "publication_date": "2024-01-15",
        "journal": "AI Research Quarterly",
        "doi": "10.1234/airq.2024.001",
        
        # Processing information
        "parser_type": "PDFParser",
        "chunk_number": 1,
        "chunk_size": 1000,
        "embedding_model": "nomic-embed-text",
        
        # Extractor results
        "extractors": {
            "entities": {
                "persons": ["John Smith", "Alice Doe"],
                "organizations": ["Stanford University"],
                "locations": ["California, USA"]
            },
            "statistics": {
                "figures": 8,
                "tables": 3,
                "equations": 15,
                "references": 45
            },
            "summary": {
                "auto_summary": "This paper presents novel architectures...",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
            }
        }
    },
    embeddings=[0.123, -0.456, 0.789, ...]  # 768-dimensional vector
)
```

## Integration with RAG Strategies

All RAG strategies should follow these metadata practices:

1. **Parser Stage**: Extract basic metadata (source, type, size)
2. **Extractor Stage**: Enhance with domain-specific metadata
3. **Embedding Stage**: Preserve all metadata alongside embeddings
4. **Storage Stage**: Convert metadata to storage-compatible format
5. **Retrieval Stage**: Use metadata for filtering and ranking
6. **Response Stage**: Include relevant metadata in context

By following these best practices, your RAG system will have rich, searchable metadata that enhances retrieval accuracy and enables sophisticated filtering and analysis capabilities.