# RAG Components Guide

Comprehensive documentation of all components in the RAG system, including parsers, extractors, embedders, stores, and retrievers.

## Table of Contents

1. [Component Architecture](#component-architecture)
2. [Parsers](#parsers)
3. [Extractors](#extractors)
4. [Embedders](#embedders)
5. [Vector Stores](#vector-stores)
6. [Retrievers](#retrievers)
7. [Creating Custom Components](#creating-custom-components)
8. [Component Configuration](#component-configuration)
9. [Best Practices](#best-practices)

## Component Architecture

### Base Component Structure

All components inherit from base classes that define common interfaces:

```python
class BaseComponent:
    """Base class for all RAG components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validate_config()
    
    def validate_config(self):
        """Validate component configuration."""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Main processing method."""
        pass
```

### Component Pipeline

```
Input Document
    ↓
[Parser] → Parse raw files into text
    ↓
[Extractors] → Extract metadata and features
    ↓
[Embedder] → Generate vector embeddings
    ↓
[Vector Store] → Store embeddings with metadata
    ↓
[Retriever] → Search and retrieve relevant documents
    ↓
Output Results
```

## Parsers

Parsers convert various file formats into processable text documents.

### TextParser

Handles plain text files with configurable chunking.

**Configuration:**
```yaml
parser:
  type: "TextParser"
  config:
    encoding: "utf-8"              # File encoding
    chunk_size: 512                # Characters per chunk
    chunk_overlap: 50              # Overlap between chunks
    preserve_formatting: true      # Keep original formatting
    min_chunk_size: 100           # Minimum chunk size
```

**Features:**
- Automatic encoding detection
- Smart chunking at sentence boundaries
- Hash-based deduplication
- Metadata extraction (file stats, encoding)

### PDFParser

Extracts text and metadata from PDF files.

**Configuration:**
```yaml
parser:
  type: "PDFParser"
  config:
    extract_images: false          # Extract embedded images
    extract_tables: true           # Extract tables
    extract_metadata: true         # Extract PDF metadata
    ocr_enabled: false            # Enable OCR for scanned PDFs
    combine_pages: true           # Combine pages into single document
    chunk_size: 1024              # Characters per chunk
    chunk_overlap: 200            # Overlap between chunks
```

**Features:**
- PyPDF2-based extraction
- Table detection and extraction
- Metadata extraction (author, title, creation date)
- OCR support for scanned documents (with Tesseract)
- Page-level or document-level processing

### HTMLParser

Parses HTML documents and web pages.

**Configuration:**
```yaml
parser:
  type: "HTMLParser"
  config:
    extract_links: true           # Extract URLs
    preserve_structure: false     # Keep HTML structure
    remove_scripts: true          # Remove JavaScript
    remove_styles: true           # Remove CSS
    extract_metadata: true        # Extract meta tags
    use_beautifulsoup: true       # Use BeautifulSoup (vs regex)
    chunk_size: 512
    chunk_overlap: 50
```

**Features:**
- BeautifulSoup and regex-based parsing
- Link extraction with categorization
- Clean text extraction
- Metadata from meta tags
- Script and style removal

### CSVParser

Processes CSV files with flexible field mapping.

**Configuration:**
```yaml
parser:
  type: "CSVParser"
  config:
    delimiter: ","                # Field delimiter
    has_header: true              # First row is header
    field_mapping:                # Map CSV fields to document fields
      content_field: "description"
      id_field: "ticket_id"
      metadata_fields:
        - "priority"
        - "category"
        - "customer"
    combine_fields: false         # Combine all fields into content
    encoding: "utf-8"
```

**Features:**
- Flexible field mapping
- Automatic type detection
- Metadata extraction from columns
- Support for various delimiters
- Header detection

### MarkdownParser

Specialized parser for Markdown documents.

**Configuration:**
```yaml
parser:
  type: "MarkdownParser"
  config:
    preserve_formatting: true     # Keep markdown formatting
    extract_code_blocks: true     # Extract code separately
    extract_links: true           # Extract URLs
    extract_images: true          # Extract image references
    chunk_by_section: true        # Chunk at heading boundaries
    min_section_size: 200
    max_section_size: 2000
```

**Features:**
- Structure-aware chunking
- Code block extraction
- Link and image reference extraction
- Heading hierarchy preservation
- Table extraction

## Extractors

Extractors analyze document content to extract metadata, entities, and features.

### EntityExtractor

Extracts named entities using NLP models.

**Configuration:**
```yaml
- type: "EntityExtractor"
  config:
    entities:                     # Entity types to extract
      - "PERSON"
      - "ORG"
      - "GPE"
      - "DATE"
      - "MONEY"
      - "PRODUCT"
    model: "en_core_web_sm"      # spaCy model
    confidence_threshold: 0.7     # Minimum confidence
    max_entities: 50              # Maximum entities per document
    include_context: true         # Include surrounding context
```

**Entity Types:**
- PERSON: People, characters
- ORG: Organizations, companies
- GPE: Geopolitical entities
- DATE: Dates, time periods
- MONEY: Monetary values
- PRODUCT: Products, services
- LOC: Locations
- EVENT: Events
- FAC: Facilities
- LAW: Laws, regulations

### KeywordExtractor

Extracts important keywords and phrases.

**Configuration:**
```yaml
- type: "KeywordExtractor"
  config:
    max_keywords: 10              # Maximum keywords
    algorithm: "yake"             # Algorithm: yake, rake, tfidf
    language: "en"                # Language for processing
    ngram_range: [1, 3]          # N-gram size range
    deduplication_threshold: 0.7  # Remove similar keywords
    include_scores: true          # Include relevance scores
```

**Algorithms:**
- **YAKE**: Statistical keyword extraction
- **RAKE**: Rapid Automatic Keyword Extraction
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **TextRank**: Graph-based ranking

### SummaryExtractor

Generates document summaries and key points.

**Configuration:**
```yaml
- type: "SummaryExtractor"
  config:
    max_sentences: 3              # Summary length in sentences
    algorithm: "extractive"       # extractive or abstractive
    include_keywords: true        # Add top keywords
    include_statistics: true      # Add document statistics
    min_sentence_length: 10       # Minimum sentence length
    language: "en"
```

**Features:**
- Extractive summarization (key sentence selection)
- Abstractive summarization (with LLMs)
- Key phrase extraction
- Statistical summary (word count, readability)

### HeadingExtractor

Extracts document structure and headings.

**Configuration:**
```yaml
- type: "HeadingExtractor"
  config:
    levels: [1, 2, 3, 4]         # Heading levels to extract
    include_content: true         # Include section content
    max_content_length: 200       # Maximum content preview
    preserve_hierarchy: true      # Maintain heading structure
    format: "markdown"            # Output format
```

**Features:**
- Multi-level heading extraction
- Section content preview
- Hierarchy preservation
- Table of contents generation

### PatternExtractor

Extracts content matching specific patterns.

**Configuration:**
```yaml
- type: "PatternExtractor"
  config:
    patterns:                     # Named patterns
      email: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      phone: "\\+?1?\\d{9,15}"
      url: "https?://[^\\s]+"
      ip_address: "\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b"
      custom_id: "ID-\\d{6}"
    include_context: true         # Include surrounding text
    context_window: 50           # Context characters
    case_sensitive: false
```

**Common Patterns:**
- Email addresses
- Phone numbers
- URLs
- IP addresses
- Social Security Numbers (with care)
- Credit card numbers (with care)
- Custom identifiers

### LinkExtractor

Extracts and categorizes links.

**Configuration:**
```yaml
- type: "LinkExtractor"
  config:
    validate_urls: false          # Validate URL accessibility
    extract_anchor_text: true     # Extract link text
    categorize: true              # Categorize as internal/external
    follow_redirects: false       # Follow HTTP redirects
    max_links: 100               # Maximum links to extract
```

**Features:**
- URL validation
- Anchor text extraction
- Internal/external categorization
- Redirect resolution
- Domain extraction

### ContentStatisticsExtractor

Calculates document statistics and metrics.

**Configuration:**
```yaml
- type: "ContentStatisticsExtractor"
  config:
    calculate_readability: true   # Readability scores
    calculate_sentiment: false    # Sentiment analysis
    word_frequency: true          # Word frequency analysis
    structural_analysis: true     # Document structure metrics
    language_detection: true      # Detect language
```

**Metrics:**
- Readability scores (Flesch-Kincaid, etc.)
- Word and sentence counts
- Average word/sentence length
- Vocabulary richness
- Language detection
- Sentiment scores

### TableExtractor

Extracts and processes tables.

**Configuration:**
```yaml
- type: "TableExtractor"
  config:
    min_rows: 2                  # Minimum table rows
    min_cols: 2                  # Minimum table columns
    extract_headers: true         # Extract column headers
    format: "markdown"            # Output format: markdown, json, csv
    max_tables: 10               # Maximum tables to extract
```

**Features:**
- Table detection
- Header extraction
- Multiple output formats
- Cell content cleaning
- Table metadata

## Embedders

Embedders convert text into vector representations.

### OllamaEmbedder

Local embeddings using Ollama.

**Configuration:**
```yaml
embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"     # Model name
    host: "http://localhost:11434" # Ollama host
    dimension: 768                # Embedding dimension
    batch_size: 32                # Batch size for processing
    timeout: 60                   # Request timeout
    normalize: true               # Normalize vectors
```

**Supported Models:**
- nomic-embed-text (768d)
- all-minilm (384d)
- Custom models

### OpenAIEmbedder (Future)

OpenAI API embeddings.

**Configuration:**
```yaml
embedder:
  type: "OpenAIEmbedder"
  config:
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimension: 1536
    batch_size: 100
    max_retries: 3
    timeout: 30
```

### HuggingFaceEmbedder (Future)

Local embeddings using HuggingFace models.

**Configuration:**
```yaml
embedder:
  type: "HuggingFaceEmbedder"
  config:
    model: "sentence-transformers/all-mpnet-base-v2"
    dimension: 768
    device: "cuda"                # cuda, cpu, mps
    batch_size: 64
    max_length: 512
    normalize: true
```

## Vector Stores

Vector stores manage embedding storage and retrieval.

### ChromaStore

Local vector database using ChromaDB.

**Configuration:**
```yaml
vector_store:
  type: "ChromaStore"
  config:
    collection_name: "my_documents"
    distance_metric: "cosine"     # cosine, l2, ip
    embedding_function: null       # Use default
    metadata_config:              # Metadata handling
      indexed_fields: ["category", "date"]
      filterable_fields: ["priority", "author"]
```

**Features:**
- Persistent storage
- Metadata filtering
- Multiple distance metrics
- Collection management
- Batch operations

### QdrantStore (Future)

High-performance vector database.

**Configuration:**
```yaml
vector_store:
  type: "QdrantStore"
  config:
    host: "localhost"
    port: 6333
    collection_name: "documents"
    vector_size: 768
    distance: "Cosine"
    on_disk: false
    hnsw_config:
      m: 16
      ef_construct: 200
```

### PineconeStore (Future)

Cloud-based vector database.

**Configuration:**
```yaml
vector_store:
  type: "PineconeStore"
  config:
    api_key: "${PINECONE_API_KEY}"
    environment: "us-west1-gcp"
    index_name: "my-index"
    dimension: 768
    metric: "cosine"
    pods: 1
    pod_type: "p1.x1"
```

## Retrievers

Retrievers implement search strategies for finding relevant documents.

### BasicSimilarityStrategy

Simple vector similarity search.

**Configuration:**
```yaml
retrieval_strategy:
  type: "BasicSimilarityStrategy"
  config:
    top_k: 5                      # Number of results
    distance_metric: "cosine"     # Distance metric
    score_threshold: 0.7          # Minimum similarity
    include_metadata: true        # Include document metadata
```

### MetadataFilteredStrategy

Filter results based on metadata.

**Configuration:**
```yaml
retrieval_strategy:
  type: "MetadataFilteredStrategy"
  config:
    top_k: 10
    filters:                      # Metadata filters
      category: "technical"
      priority: ["high", "critical"]
      date_after: "2024-01-01"
      author: "John Doe"
    filter_mode: "pre"            # pre or post filtering
    operator: "AND"               # AND or OR
```

**Filter Operators:**
- Equality: `field: value`
- In: `field: [value1, value2]`
- Range: `field_gte: value`, `field_lte: value`
- Exists: `field_exists: true`
- Pattern: `field_regex: "pattern"`

### RerankedStrategy

Re-rank results using multiple factors.

**Configuration:**
```yaml
retrieval_strategy:
  type: "RerankedStrategy"
  config:
    initial_k: 20                 # Initial retrieval
    final_k: 5                    # Final results
    rerank_factors:
      recency_weight: 0.2         # Date-based boost
      length_weight: 0.1          # Length preference
      relevance_weight: 0.5       # Similarity score
      metadata_boost:             # Metadata-based boosts
        priority:
          critical: 2.0
          high: 1.5
          medium: 1.0
          low: 0.8
```

### MultiQueryStrategy

Generate multiple query variations.

**Configuration:**
```yaml
retrieval_strategy:
  type: "MultiQueryStrategy"
  config:
    num_queries: 3                # Number of variations
    query_generator: "synonym"    # Generation method
    aggregation: "rrf"            # Aggregation method
    top_k_per_query: 5
    final_k: 5
    diversity_threshold: 0.3      # Result diversity
```

**Aggregation Methods:**
- **RRF**: Reciprocal Rank Fusion
- **Max**: Maximum score
- **Mean**: Average score
- **Weighted**: Weighted combination

### HybridUniversalStrategy

Combine multiple strategies.

**Configuration:**
```yaml
retrieval_strategy:
  type: "HybridUniversalStrategy"
  config:
    strategies:
      - type: "BasicSimilarityStrategy"
        weight: 0.6
        config:
          top_k: 10
      - type: "MetadataFilteredStrategy"
        weight: 0.4
        config:
          top_k: 10
          filters:
            category: "important"
    fusion_method: "weighted"     # weighted or rrf
    final_k: 5
```

## Creating Custom Components

### Custom Parser Example

```python
from typing import List, Dict, Any
from pathlib import Path
from components.parsers.base import BaseParser
from core.document import Document

class CustomFormatParser(BaseParser):
    """Parser for custom file format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.encoding = config.get("encoding", "utf-8")
        self.chunk_size = config.get("chunk_size", 512)
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file."""
        return file_path.suffix == ".custom"
    
    def parse(self, file_path: Path) -> List[Document]:
        """Parse custom format file."""
        content = file_path.read_text(encoding=self.encoding)
        
        # Custom parsing logic
        parsed_data = self._parse_custom_format(content)
        
        # Create documents
        documents = []
        for chunk in self._chunk_content(parsed_data):
            doc = Document(
                content=chunk["text"],
                metadata={
                    "source": str(file_path),
                    "format": "custom",
                    **chunk.get("metadata", {})
                }
            )
            documents.append(doc)
        
        return documents
    
    def _parse_custom_format(self, content: str) -> Dict:
        """Parse custom format logic."""
        # Your parsing logic here
        pass
```

### Custom Extractor Example

```python
from typing import Dict, List, Any
from components.extractors.base import BaseExtractor
from core.document import Document

class SentimentExtractor(BaseExtractor):
    """Extract sentiment from documents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self._load_model(config.get("model", "textblob"))
        self.granularity = config.get("granularity", "document")
    
    def extract(self, documents: List[Document]) -> List[Document]:
        """Extract sentiment from documents."""
        for doc in documents:
            if self.granularity == "document":
                sentiment = self._analyze_sentiment(doc.content)
                doc.metadata["sentiment"] = sentiment
            elif self.granularity == "sentence":
                sentences = self._split_sentences(doc.content)
                sentiments = [self._analyze_sentiment(s) for s in sentences]
                doc.metadata["sentence_sentiments"] = sentiments
                doc.metadata["average_sentiment"] = np.mean(sentiments)
        
        return documents
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        # Your sentiment analysis logic
        return {
            "polarity": 0.5,  # -1 to 1
            "subjectivity": 0.3,  # 0 to 1
            "confidence": 0.8
        }
```

### Custom Retriever Example

```python
from typing import List, Dict, Any
from components.retrievers.base import BaseRetriever
from core.document import Document

class SemanticReranker(BaseRetriever):
    """Rerank results using semantic similarity."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self._load_reranking_model(
            config.get("model", "cross-encoder")
        )
        self.top_k = config.get("top_k", 5)
    
    def retrieve(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on semantic similarity."""
        # Score each document
        scores = []
        for doc in documents:
            score = self.model.score(query, doc.content)
            scores.append(score)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:self.top_k]
        
        # Return top-k documents
        return [documents[i] for i in ranked_indices]
```

## Component Configuration

### Configuration Inheritance

Components can inherit and override configurations:

```yaml
# Base configuration
base_config: &base
  chunk_size: 512
  chunk_overlap: 50

# Component with inherited config
parser:
  type: "TextParser"
  config:
    <<: *base
    encoding: "utf-8"  # Additional config
```

### Environment Variables

Use environment variables for sensitive data:

```yaml
embedder:
  type: "OpenAIEmbedder"
  config:
    api_key: "${OPENAI_API_KEY}"  # From environment
    model: "text-embedding-3-small"
```

### Dynamic Configuration

Configure components based on runtime conditions:

```python
def get_parser_config(file_size: int) -> Dict:
    """Get parser configuration based on file size."""
    if file_size < 1_000_000:  # < 1MB
        return {
            "chunk_size": 512,
            "chunk_overlap": 50
        }
    elif file_size < 10_000_000:  # < 10MB
        return {
            "chunk_size": 1024,
            "chunk_overlap": 100
        }
    else:  # Large files
        return {
            "chunk_size": 2048,
            "chunk_overlap": 200,
            "streaming": True
        }
```

## Best Practices

### 1. Component Selection

Choose components based on your use case:

| Use Case | Parser | Extractors | Embedder | Retriever |
|----------|--------|------------|----------|-----------|
| Academic | PDFParser | Citation, Entity | Large model | Reranked |
| Support | CSVParser | Pattern, Sentiment | Fast model | Filtered |
| Technical | MarkdownParser | Heading, Code | Technical model | Hybrid |
| News | HTMLParser | Entity, Summary | Multilingual | Multi-query |

### 2. Configuration Optimization

**For Speed:**
```yaml
# Fast configuration
parser:
  config:
    chunk_size: 256  # Smaller chunks
embedder:
  config:
    model: "all-MiniLM-L6-v2"  # Fast model
    batch_size: 64  # Larger batches
retrieval_strategy:
  type: "BasicSimilarityStrategy"  # Simple strategy
```

**For Quality:**
```yaml
# Quality configuration
parser:
  config:
    chunk_size: 1024  # Larger context
    chunk_overlap: 200  # More overlap
embedder:
  config:
    model: "bge-large-en"  # Better model
extractors:
  - type: "EntityExtractor"
  - type: "SummaryExtractor"
  - type: "KeywordExtractor"
retrieval_strategy:
  type: "RerankedStrategy"  # Advanced reranking
```

### 3. Error Handling

Implement robust error handling:

```python
class RobustParser(BaseParser):
    def parse(self, file_path: Path) -> List[Document]:
        try:
            return self._parse_impl(file_path)
        except UnicodeDecodeError:
            # Try different encoding
            return self._parse_with_fallback_encoding(file_path)
        except Exception as e:
            # Log error and return empty
            logger.error(f"Failed to parse {file_path}: {e}")
            return []
```

### 4. Performance Monitoring

Track component performance:

```python
from time import time

class MonitoredComponent:
    def process(self, *args, **kwargs):
        start = time()
        try:
            result = self._process_impl(*args, **kwargs)
            self.record_metric("success", time() - start)
            return result
        except Exception as e:
            self.record_metric("failure", time() - start)
            raise
```

### 5. Testing Components

Test components thoroughly:

```python
def test_parser():
    parser = TextParser({"chunk_size": 100})
    
    # Test basic parsing
    docs = parser.parse(Path("test.txt"))
    assert len(docs) > 0
    assert all(len(d.content) <= 100 for d in docs)
    
    # Test edge cases
    assert parser.parse(Path("empty.txt")) == []
    assert parser.parse(Path("large.txt"))  # Should not crash
```

## Related Documentation

- [Strategy System](STRATEGY_SYSTEM.md) - Configure component pipelines
- [CLI Guide](../cli/README.md) - Use components via CLI
- [API Reference](API_REFERENCE.md) - Programmatic component usage
- [Demos](../demos/README.md) - Components in action