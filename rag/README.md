# RAG System - Developer Quick Start Guide

A powerful, extensible RAG (Retrieval-Augmented Generation) system featuring **strategy-first configuration** and modular architecture. Built for developers who want to get started quickly and extend easily.

## 🚀 Quick Start for Developers

### Prerequisites
- **Python 3.8+**
- **UV** (fast Python package manager) - [Installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** (for local embeddings) - [Download](https://ollama.com/download)

### 1. Installation (1 minute)

```bash
# Clone and setup
cd rag/
uv sync                    # Install all dependencies
ollama serve               # Start Ollama (in separate terminal)
ollama pull nomic-embed-text  # Download embedding model
```

### 2. Run Your First Demo (30 seconds)

```bash
# Option 1: Single impressive demo
uv run python demos/enhanced_demo.py

# Option 2: All 5 domain-specific demos (15-25 minutes)
uv run python demos/master_demo.py

# Option 3: Individual domain demos
uv run python demos/demo1_research_papers.py
uv run python demos/demo2_customer_support.py
uv run python demos/demo3_code_documentation.py
uv run python demos/demo4_news_analysis.py
uv run python demos/demo5_business_reports.py
```

### 3. Master the CLI (Comprehensive Guide)

```bash
# List available strategies and get recommendations
uv run python cli.py strategies list
uv run python cli.py strategies recommend --use-case customer_support

# Quick start with your data
uv run python cli.py --strategy simple ingest your_documents.pdf
uv run python cli.py --strategy simple search "your query here"

# Try different strategies
uv run python cli.py --strategy customer_support ingest support_data.csv
uv run python cli.py --strategy legal ingest legal_docs/
uv run python cli.py --strategy research ingest research_papers/
```

### 4. Run Tests (Verify Everything Works)

```bash
# Quick validation
uv run python -m pytest tests/ -v

# Comprehensive component testing
uv run python tests/test_new_parsers.py
uv run python tests/test_new_extractors.py  
uv run python tests/test_new_embedders.py
uv run python tests/test_new_stores.py

# All tests (215+ tests)
uv run pytest tests/
```

## 🎯 Key Features for Developers

### Strategy-First Architecture
- **8+ Predefined Strategies**: `simple`, `customer_support`, `legal`, `research`, `news_analysis`, etc.
- **One-Command Setup**: `--strategy simple` gets you running immediately
- **Easy Customization**: Override specific settings while keeping strategy benefits
- **Smart Recommendations**: CLI suggests strategies based on your use case

### Modular Component System
- **8+ Parsers**: CSV, PDF, Markdown, Word, Excel, HTML, Plain Text
- **11+ Extractors**: Keywords, entities, tables, links, headings, summaries (all local, no APIs needed)
- **4+ Embedders**: Ollama, OpenAI, HuggingFace, SentenceTransformers
- **4+ Vector Stores**: ChromaDB, FAISS, Pinecone, Qdrant
- **5+ Retrieval Strategies**: From basic similarity to advanced hybrid approaches

### Developer Experience
- **Factory Pattern**: Easy component registration and creation
- **Comprehensive Schema**: Complete API documentation in `schema.yaml`
- **Full Test Coverage**: 215+ tests passing, comprehensive mocking
- **Rich CLI**: Beautiful terminal output with progress bars and tables
- **Extensible**: Add new components without changing core code

## 📚 Demo System (Learn by Example)

The RAG system includes 5 comprehensive demos showing different strategies and real-world applications:

| Demo | Domain | What It Shows | Runtime |
|------|--------|---------------|---------|
| **Research Papers** | Academia | Statistical analysis + entity extraction | 3-5 min |
| **Customer Support** | Business | Case matching + pattern recognition | 2-4 min |
| **Code Documentation** | Technical | Structure preservation + cross-references | 2-3 min |
| **News Analysis** | Media | Sentiment analysis + trend tracking | 3-4 min |
| **Business Reports** | Finance | Tabular data + financial metrics extraction | 3-5 min |

### Run Enhanced Demo (Best Starting Point)

```bash
uv run python demos/enhanced_demo.py
```

This demo showcases:
- Document ingestion from mixed formats
- Metadata extraction and enrichment  
- Pattern-based entity recognition
- Vector similarity search
- Metadata-filtered retrieval
- Multi-query expansion
- Reranked results with custom scoring

## 📊 Component Library

### 📄 Document Parsers

Complete list of available parsers for document ingestion:

| Parser | File | Formats | Dependencies | When to Use | Example Usage |
|--------|------|---------|--------------|-------------|---------------|
| **PlainTextParser** | [`text_parser.py`](components/parsers/text_parser/text_parser.py) | `.txt`, `.log`, `.md` | None | Simple text files, logs, READMEs | `PlainTextParser(name="text", config={"chunk_size": 512})` |
| **CSVParser** | [`csv_parser.py`](components/parsers/csv_parser/csv_parser.py) | `.csv`, `.tsv` | pandas (optional) | Structured tabular data, databases exports | `CSVParser(name="csv", config={"delimiter": ","})` |
| **PDFParser** | [`pdf_parser.py`](components/parsers/pdf_parser/pdf_parser.py) | `.pdf` | PyMuPDF | Documents, reports, papers | `PDFParser(name="pdf", config={"combine_pages": true})` |
| **MarkdownParser** | [`markdown_parser.py`](components/parsers/markdown_parser/markdown_parser.py) | `.md`, `.markdown` | markdown2 | Documentation, READMEs, notes | `MarkdownParser(name="md", config={"extract_headers": true})` |
| **HTMLParser** | [`html_parser.py`](components/parsers/html_parser/html_parser.py) | `.html`, `.htm` | BeautifulSoup4 | Web pages, scraped content | `HTMLParser(name="html", config={"extract_links": true})` |
| **DocxParser** | [`docx_parser.py`](components/parsers/docx_parser/docx_parser.py) | `.docx` | python-docx | Word documents, reports | `DocxParser(name="docx", config={"extract_tables": true})` |
| **ExcelParser** | [`excel_parser.py`](components/parsers/excel_parser/excel_parser.py) | `.xlsx`, `.xls` | pandas, openpyxl | Spreadsheets, financial data | `ExcelParser(name="excel", config={"parse_all_sheets": true})` |
| **CustomerSupportCSVParser** | [`csv_parser.py`](parsers/csv_parser.py) | `.csv` | None | Support tickets, CRM exports | Specialized for customer support data |

#### Parser Configuration Examples

```python
# Basic text parsing
parser = PlainTextParser(name="text_parser", config={
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "encoding": "utf-8",
    "preserve_line_breaks": True
})

# PDF with metadata extraction
parser = PDFParser(name="pdf_parser", config={
    "combine_pages": False,  # Each page as separate document
    "extract_metadata": True,
    "include_page_numbers": True,
    "min_text_length": 50
})

# Markdown with structure preservation
parser = MarkdownParser(name="md_parser", config={
    "extract_frontmatter": True,
    "chunk_by_headings": True,
    "preserve_code_blocks": True,
    "extract_links": True
})
```

### 🔍 Metadata Extractors

Components for enriching documents with additional metadata:

| Extractor | File | Extracts | Dependencies | When to Use | Example Usage |
|-----------|------|----------|--------------|-------------|---------------|
| **PathExtractor** | [`path_extractor.py`](components/extractors/path_extractor/path_extractor.py) | File paths, directories, extensions | None | File organization, categorization | `PathExtractor("path", {"store_full_path": true})` |
| **KeywordExtractor** | [`keyword_extractor.py`](components/extractors/keyword_extractor/keyword_extractor.py) | Keywords, key phrases | YAKE, spaCy (optional) | SEO, tagging, summarization | `KeywordExtractor("keywords", {"method": "yake", "max_keywords": 10})` |
| **EntityExtractor** | [`entity_extractor.py`](components/extractors/entity_extractor/entity_extractor.py) | Named entities (people, orgs, locations) | spaCy | Information extraction, analytics | `EntityExtractor("entities", {"model": "en_core_web_sm"})` |
| **LinkExtractor** | [`link_extractor.py`](components/extractors/link_extractor/link_extractor.py) | URLs, email addresses | None | Web scraping, reference tracking | `LinkExtractor("links", {"include_external": true})` |
| **HeadingExtractor** | [`heading_extractor.py`](components/extractors/heading_extractor/heading_extractor.py) | Document headings, structure | None | Navigation, TOC generation | `HeadingExtractor("headings", {"extract_hierarchy": true})` |
| **TableExtractor** | [`table_extractor.py`](components/extractors/table_extractor/table_extractor.py) | Tables, structured data | pandas (optional) | Data analysis, reporting | `TableExtractor("tables", {"format": "markdown"})` |
| **DateTimeExtractor** | [`datetime_extractor.py`](components/extractors/datetime_extractor/datetime_extractor.py) | Dates, times, durations | dateutil | Timeline analysis, scheduling | `DateTimeExtractor("dates", {"extract_relative": true})` |
| **StatisticsExtractor** | [`statistics_extractor.py`](components/extractors/statistics_extractor/statistics_extractor.py) | Word count, readability scores | textstat (optional) | Content analysis, complexity | `StatisticsExtractor("stats", {"calculate_readability": true})` |
| **SummaryExtractor** | [`summary_extractor.py`](components/extractors/summary_extractor/summary_extractor.py) | Text summaries, abstracts | sumy, transformers (optional) | Summarization, preview generation | `SummaryExtractor("summary", {"method": "textrank", "sentences": 3})` |
| **PatternExtractor** | [`pattern_extractor.py`](components/extractors/pattern_extractor/pattern_extractor.py) | Custom regex patterns | None | Domain-specific extraction | `PatternExtractor("patterns", {"patterns": {"email": r"\S+@\S+"}})` |

#### Extractor Configuration Examples

```python
# Extract keywords with YAKE
extractor = KeywordExtractor("keyword_extractor", config={
    "method": "yake",
    "language": "en",
    "max_keywords": 15,
    "deduplication_threshold": 0.7
})

# Extract named entities
extractor = EntityExtractor("entity_extractor", config={
    "model": "en_core_web_lg",
    "entity_types": ["PERSON", "ORG", "GPE", "DATE"],
    "confidence_threshold": 0.8
})

# Extract custom patterns
extractor = PatternExtractor("pattern_extractor", config={
    "patterns": {
        "phone": r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    }
})
```

### 🧠 Embedding Models

Components for generating vector embeddings:

| Embedder | File | Models | API/Local | When to Use | Example Usage |
|----------|------|--------|-----------|-------------|---------------|
| **OllamaEmbedder** | [`ollama_embedder.py`](components/embedders/ollama_embedder/ollama_embedder.py) | nomic-embed-text, mxbai-embed-large | Local | Privacy-first, offline usage | `OllamaEmbedder(name="ollama", config={"model": "nomic-embed-text"})` |
| **OpenAIEmbedder** | [`openai_embedder.py`](components/embedders/openai_embedder/openai_embedder.py) | text-embedding-3-small/large | API | High quality, cloud-based | `OpenAIEmbedder(name="openai", config={"model": "text-embedding-3-small"})` |
| **HuggingFaceEmbedder** | [`huggingface_embedder.py`](components/embedders/huggingface_embedder/huggingface_embedder.py) | Any HF model | Local/API | Custom models, research | `HuggingFaceEmbedder(name="hf", config={"model": "BAAI/bge-small-en"})` |
| **SentenceTransformerEmbedder** | [`sentence_transformer_embedder.py`](components/embedders/sentence_transformer_embedder/sentence_transformer_embedder.py) | all-MiniLM-L6-v2, etc. | Local | Fast, efficient, multilingual | `SentenceTransformerEmbedder(name="st", config={"model": "all-mpnet-base-v2"})` |

#### Embedder Configuration Examples

```python
# Local embeddings with Ollama
embedder = OllamaEmbedder(name="ollama_embedder", config={
    "model": "nomic-embed-text",
    "api_base": "http://localhost:11434",
    "batch_size": 32,
    "timeout": 30
})

# OpenAI embeddings
embedder = OpenAIEmbedder(name="openai_embedder", config={
    "model": "text-embedding-3-small",
    "api_key": "${OPENAI_API_KEY}",  # From environment
    "dimensions": 1536,
    "batch_size": 100
})

# Sentence Transformers
embedder = SentenceTransformerEmbedder(name="st_embedder", config={
    "model": "all-MiniLM-L6-v2",
    "device": "cuda",  # or "cpu"
    "normalize_embeddings": True,
    "batch_size": 64
})
```

### 🗄️ Vector Stores

Components for storing and retrieving embeddings:

| Store | File | Type | Scale | When to Use | Example Usage |
|-------|------|------|-------|-------------|---------------|
| **ChromaStore** | [`chroma_store.py`](components/stores/chroma_store.py) | Local/Server | <1M docs | Prototyping, small-medium datasets | `ChromaStore(name="chroma", config={"persist_directory": "./chroma_db"})` |
| **FAISSStore** | [`faiss_store.py`](components/stores/faiss_store.py) | Local | Any size | High-performance similarity search | `FAISSStore(name="faiss", config={"index_type": "IVF"})` |
| **PineconeStore** | [`pinecone_store.py`](components/stores/pinecone_store.py) | Cloud | Any size | Managed service, production | `PineconeStore(name="pinecone", config={"api_key": "..."})` |
| **QdrantStore** | [`qdrant_store.py`](components/stores/qdrant_store.py) | Local/Cloud | >1M docs | Production, advanced filtering | `QdrantStore(name="qdrant", config={"host": "localhost"})` |

#### Vector Store Configuration Examples

```python
# ChromaDB for local development
store = ChromaStore(name="chroma_store", config={
    "persist_directory": "./chroma_db",
    "collection_name": "documents",
    "distance_metric": "cosine"
})

# FAISS for high-performance
store = FAISSStore(name="faiss_store", config={
    "index_type": "IVF",
    "nlist": 100,
    "nprobe": 10,
    "use_gpu": False
})

# Qdrant for production
store = QdrantStore(name="qdrant_store", config={
    "host": "localhost",
    "port": 6333,
    "collection_name": "documents",
    "vector_size": 768,
    "distance": "Cosine"
})
```

### 🎯 Retrieval Strategies

Advanced retrieval strategies for different use cases:

| Strategy | File | Description | When to Use | Example Usage |
|----------|------|-------------|-------------|---------------|
| **BasicSimilarityStrategy** | [`retrieval/strategies/`](retrieval/strategies/universal/basic_similarity.py) | Simple cosine similarity | Quick prototypes, baseline | `{"type": "BasicSimilarityStrategy", "config": {"top_k": 10}}` |
| **MetadataFilteredStrategy** | [`retrieval/strategies/`](retrieval/strategies/universal/metadata_filtered.py) | Filter by metadata before search | Structured data, categories | `{"type": "MetadataFilteredStrategy", "config": {"filters": {...}}}` |
| **MultiQueryStrategy** | [`retrieval/strategies/`](retrieval/strategies/universal/multi_query.py) | Generate multiple query variants | Improve recall, query expansion | `{"type": "MultiQueryStrategy", "config": {"num_variants": 3}}` |
| **RerankedStrategy** | [`retrieval/strategies/`](retrieval/strategies/universal/reranked.py) | Re-rank with multiple factors | Precision improvement | `{"type": "RerankedStrategy", "config": {"rerank_factors": {...}}}` |
| **HybridUniversalStrategy** | [`retrieval/strategies/`](retrieval/strategies/universal/hybrid_universal.py) | Combine multiple strategies | Best of all approaches | `{"type": "HybridUniversalStrategy", "config": {"strategies": [...]}}` |

## 🛠️ Advanced CLI Usage

### Strategy Operations

```bash
# List all available strategies with descriptions
uv run python cli.py strategies list

# Get detailed information about a strategy
uv run python cli.py strategies show customer_support --detailed

# Get strategy recommendations for your use case
uv run python cli.py strategies recommend --use-case "analyzing legal contracts"

# Test a strategy configuration
uv run python cli.py strategies test simple --sample-file test.pdf

# Convert traditional config to strategy
uv run python cli.py strategies convert legal config/legal.yaml
```

### Data Ingestion

```bash
# Ingest single file
uv run python cli.py --strategy simple ingest document.pdf

# Ingest directory
uv run python cli.py --strategy research ingest papers/ --recursive

# Ingest with metadata
uv run python cli.py --strategy simple ingest data.csv \
  --metadata '{"source": "internal", "department": "sales"}'

# Ingest with custom configuration overrides
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"parser":{"config":{"chunk_size":256}}}}' \
  ingest documents/
```

### Search Operations

```bash
# Basic search
uv run python cli.py --strategy simple search "your query"

# Search with filters
uv run python cli.py --strategy simple search "error logs" \
  --filters '{"file_type": "log", "date": {"$gte": "2024-01-01"}}'

# Search with custom top_k
uv run python cli.py --strategy simple search "machine learning" --top-k 20

# Search with specific retrieval strategy
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"retrieval_strategy":{"type":"RerankedStrategy"}}}' \
  search "important document"
```

### Performance Optimization

```bash
# Memory-constrained environment
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":4}}}}' \
  ingest documents/

# Speed-optimized search (disable complex retrievers)
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"retrieval_strategy":{"type":"BasicSimilarityStrategy"}}}' \
  search "quick query"
```

### Debugging and Development

#### 1. Verbose Output and Debugging
```bash
# Enable debug logging
uv run python cli.py --log-level DEBUG --strategy simple ingest test.csv

# Detailed strategy information
uv run python cli.py strategies show production --detailed

# Test strategy with verbose output
uv run python cli.py --log-level INFO strategies test customer_support --sample-file test_tickets.csv
```

#### 2. Configuration Validation
```bash
# Validate strategy configuration
uv run python cli.py strategies test simple

# Test overrides syntax
uv run python cli.py strategies test customer_support \
  --overrides '{"components":{"embedder":{"config":{"batch_size":16}}}}'

# Convert and validate traditional config
uv run python cli.py strategies convert legal legal_config.yaml
uv run python cli.py --config legal_config.yaml test
```

## 🛠️ Development Commands

### Everyday Development
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Run with UV (recommended)
uv run python cli.py --help
uv run python your_script.py

# Traditional activation (alternative)
source .venv/bin/activate
python cli.py --help
```

### Testing & Quality
```bash
# Run all tests (215+ tests)
uv run python -m pytest tests/

# Test specific components
uv run python -m pytest tests/components/parsers/
uv run python -m pytest tests/components/extractors/

# Test with coverage
uv run python -m pytest --cov=components --cov-report=html

# Integration tests
uv run python tests/test_retrieval_system.py
```

### Code Quality
```bash
# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Linting
uv run ruff check .
```

## 🏗️ Architecture Overview

The system uses a **modular, strategy-first architecture**:

```
📦 Components (Extensible)
├── 📄 Parsers: PlainTextParser, CSVParser, PDFParser, MarkdownParser...
├── 🔍 Extractors: YAKEExtractor, EntityExtractor, TableExtractor...
├── 🧠 Embedders: OllamaEmbedder, OpenAIEmbedder, HuggingFaceEmbedder...
├── 🗄️ Stores: ChromaStore, FAISSStore, PineconeStore, QdrantStore...
└── 🎯 Retrievers: BasicSimilarity, MetadataFiltered, MultiQuery, Hybrid...

🏭 Core System
├── 🏗️ Factories: Component creation and registration
├── 📋 Strategies: Pre-configured combinations for common use cases
├── 🔄 Pipeline: Document processing workflow
└── 📊 Schema: Configuration validation and API documentation

🎭 User Interface
├── 🖥️ CLI: Full-featured command-line interface
├── 🎬 Demos: 5 comprehensive domain-specific showcases
└── 🧪 Tests: 215+ tests with mocking and validation
```

### Key Patterns

1. **Factory Pattern**: All components register with factories for dynamic creation
2. **Strategy System**: Pre-configured combinations optimized for specific use cases
3. **Configuration-Driven**: Everything configurable via YAML with Pydantic validation
4. **Local-First**: Prioritize local execution with optional cloud providers

## 🔧 Adding New Components

The system is designed for easy extension. Here's how to add new components:

### 1. Add a New Parser

```bash
# Create structure
mkdir -p components/parsers/your_parser/
touch components/parsers/your_parser/{__init__.py,your_parser.py,schema.py}
```

```python
# components/parsers/your_parser/your_parser.py
from typing import List, Dict, Any
from core.base import BaseParser, Document

class YourParser(BaseParser):
    """Parser for your specific format."""
    
    def parse(self, content: bytes, metadata: Dict[str, Any] = None) -> List[Document]:
        # Your parsing logic here
        return [Document(content=text, metadata=metadata or {})]
```

```python
# Register in core/factories.py
from components.parsers.your_parser.your_parser import YourParser

class ParserFactory(ComponentFactory):
    _registry = {
        # ... existing parsers
        "YourParser": YourParser,
    }
```

### 2. Add Tests

```python
# tests/components/parsers/test_your_parser.py
import pytest
from components.parsers.your_parser.your_parser import YourParser

class TestYourParser:
    def test_basic_parsing(self):
        parser = YourParser(name="YourParser", config={})
        # Test your parsing logic
```

### 3. Add to Strategy (Optional)

```yaml
# strategies/your_strategy.yaml
your_strategy:
  description: "Optimized for your use case"
  components:
    parser:
      type: "YourParser"
      config:
        option1: true
```

The same pattern applies to **Extractors**, **Embedders**, **Stores**, and **Retrievers**. See [`STRUCTURE.md`](STRUCTURE.md) for detailed developer guidance.

## 📖 Configuration Examples

### Strategy-Based (Recommended)
```bash
# Use predefined strategies
uv run python cli.py --strategy simple ingest data.csv
uv run python cli.py --strategy customer_support ingest tickets.csv
uv run python cli.py --strategy legal ingest contracts/

# Customize strategy settings
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":32}}}}' \
  ingest data/
```

### Traditional Configuration
```yaml
# config/custom.yaml
version: "v1"
parser:
  type: "PDFParser"
  config:
    combine_pages: true
    extract_metadata: true

embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"
    batch_size: 32

vector_store:
  type: "ChromaStore"
  config:
    persist_directory: "./chroma_db"
    collection_name: "documents"

retrieval_strategy:
  type: "HybridUniversalStrategy"
  config:
    strategies:
      - type: "BasicSimilarityStrategy"
        weight: 0.5
      - type: "MetadataFilteredStrategy"
        weight: 0.5
```

## 🚨 Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| **Ollama not running** | Start with `ollama serve` in a separate terminal |
| **Model not found** | Pull the model: `ollama pull nomic-embed-text` |
| **Import errors** | Run `uv sync` to install all dependencies |
| **ChromaDB errors** | Delete `./chroma_db` directory and retry |
| **PDF parsing fails** | Install: `uv add pymupdf` |
| **Entity extraction fails** | Install spaCy model: `python -m spacy download en_core_web_sm` |

### Performance Tips

1. **Batch Processing**: Use larger batch sizes for embeddings (32-128)
2. **Chunking**: Adjust chunk size based on your use case (256-2048)
3. **Caching**: Enable caching for repeated queries
4. **GPU**: Use GPU for embeddings when available
5. **Indexing**: Use appropriate index types (IVF for large datasets)

## 📚 Documentation

- **[STRUCTURE.md](STRUCTURE.md)** - Detailed codebase structure and patterns
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Comprehensive setup instructions
- **[CLAUDE.md](CLAUDE.md)** - AI assistant integration guide
- **[schema.yaml](schema.yaml)** - Complete API schema documentation
- **[strategies/README.md](strategies/README.md)** - Strategy system documentation

## 🤝 Contributing

We welcome contributions! To add new components:

1. Follow the patterns in existing components
2. Add comprehensive tests
3. Update documentation
4. Submit a pull request

See [STRUCTURE.md](STRUCTURE.md) for detailed contribution guidelines.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- **Ollama** for local embeddings
- **ChromaDB** for vector storage
- **Rich** for beautiful CLI output
- **Pydantic** for configuration validation
- **PyMuPDF** for PDF parsing
- **BeautifulSoup4** for HTML parsing
- **YAKE** for keyword extraction

---

**Ready to build?** Start with `uv run python demos/enhanced_demo.py` to see the full system in action!