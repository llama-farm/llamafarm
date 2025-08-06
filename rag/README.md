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
```

## 🎯 Key Features for Developers

### Strategy-First Architecture
- **8+ Predefined Strategies**: `simple`, `customer_support`, `legal`, `research`, `news_analysis`, etc.
- **One-Command Setup**: `--strategy simple` gets you running immediately
- **Easy Customization**: Override specific settings while keeping strategy benefits
- **Smart Recommendations**: CLI suggests strategies based on your use case

### Modular Component System
- **12+ Parsers**: CSV, PDF, Markdown, Word, Excel, HTML, Plain Text
- **9+ Extractors**: Keywords, entities, tables, links, headings, summaries (all local, no APIs needed)
- **4+ Embedders**: Ollama, OpenAI, HuggingFace, SentenceTransformers
- **4+ Vector Stores**: ChromaDB, FAISS, Pinecone, Qdrant
- **5+ Retrieval Strategies**: From basic similarity to advanced hybrid approaches

### Developer Experience
- **Factory Pattern**: Easy component registration and creation
- **Comprehensive Schema**: Complete API documentation in `schema.yaml`
- **Full Test Coverage**: 118 tests passing, comprehensive mocking
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
| **Business Reports** | Finance | Multi-format processing + metrics extraction | 4-6 min |

Each demo:
- ✅ Processes real documents with multiple extractors
- ✅ Creates domain-specific vector databases
- ✅ Demonstrates semantic search with actual similarity scores
- ✅ Shows metadata extraction and filtering
- ✅ Explains why each strategy works for its domain

### Running Demos

```bash
# All demos with beautiful analysis (recommended first time)
uv run python demos/master_demo.py

# Quick enhanced demo showing the system in action
uv run python demos/enhanced_demo.py

# Individual domain demos
uv run python demos/demo1_research_papers.py     # Academic content
uv run python demos/demo2_customer_support.py    # Support tickets
uv run python demos/demo3_code_documentation.py  # Technical docs
uv run python demos/demo4_news_analysis.py       # Media content
uv run python demos/demo5_business_reports.py    # Financial data
```

After running demos, your vector databases are ready for querying:

```bash
# Query demo databases
uv run python cli.py search "transformer architecture" --collection research_papers
uv run python cli.py search "login issues" --collection customer_support
uv run python cli.py search "API security best practices" --collection code_documentation
```

## 🖥️ Complete CLI Reference Guide

The RAG CLI provides powerful commands for document processing, search, and system management. Here's everything you need to know:

### Global Options (Available on All Commands)

```bash
# Configuration options
--config, -c           # Path to config file (alternative to --strategy)
--strategy             # Use predefined strategy (recommended)
--strategy-overrides   # JSON overrides for strategy settings
--base-dir, -b         # Base directory for relative paths
--log-level           # Logging level (DEBUG, INFO, WARNING, ERROR)

# Examples
uv run python cli.py --strategy simple ingest data.csv
uv run python cli.py --config my_config.yaml search "query"
uv run python cli.py --log-level DEBUG --strategy research ingest papers/
```

### Strategy System (The Easy Way)

#### 1. Discover Available Strategies
```bash
# List all available strategies
uv run python cli.py strategies list

# Get detailed strategy information
uv run python cli.py strategies list --detailed

# Show specific strategy details
uv run python cli.py strategies show simple
uv run python cli.py strategies show customer_support

# Get strategy recommendations
uv run python cli.py strategies recommend --use-case customer_support
uv run python cli.py strategies recommend --performance speed --resources low
uv run python cli.py strategies recommend --use-case legal --performance accuracy
```

#### 2. Available Predefined Strategies
- **`simple`**: Basic document processing, great for getting started
- **`customer_support`**: Optimized for support tickets with entity extraction
- **`legal`**: Legal document processing with compliance features
- **`research`**: Academic papers with statistical analysis
- **`business`**: Financial reports with metrics extraction  
- **`technical`**: Code documentation with structure preservation
- **`production`**: Enterprise-ready configuration with performance optimization

#### 3. Using Strategies (Recommended Approach)
```bash
# Basic strategy usage
uv run python cli.py --strategy simple ingest documents.csv
uv run python cli.py --strategy simple search "password reset"

# Domain-specific strategies
uv run python cli.py --strategy customer_support ingest support_tickets.csv
uv run python cli.py --strategy legal ingest contracts/*.pdf
uv run python cli.py --strategy research ingest papers/*.pdf
uv run python cli.py --strategy business ingest reports/*.xlsx

# Production usage
uv run python cli.py --strategy production ingest large_dataset/
uv run python cli.py --strategy production search "complex business query"
```

#### 4. Strategy Overrides (Customize While Keeping Benefits)
```bash
# Override embedder batch size
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":32}}}}' \
  ingest data.csv

# Override vector store collection name
uv run python cli.py --strategy customer_support \
  --strategy-overrides '{"components":{"vector_store":{"config":{"collection_name":"my_tickets"}}}}' \
  ingest tickets.csv

# Override multiple components
uv run python cli.py --strategy research \
  --strategy-overrides '{
    "components": {
      "embedder": {"config": {"batch_size": 64}},
      "vector_store": {"config": {"persist_directory": "./my_research_db"}},
      "retrieval_strategy": {"config": {"top_k": 15}}
    }
  }' ingest papers/

# Complex override example - change parser and add extractors
uv run python cli.py --strategy simple \
  --strategy-overrides '{
    "components": {
      "parser": {"type": "PlainTextParser", "config": {"chunk_size": 500}},
      "extractors": [
        {"type": "YAKEExtractor", "config": {"max_keywords": 20}},
        {"type": "EntityExtractor", "config": {"entity_types": ["PERSON", "ORG"]}}
      ]
    }
  }' ingest documents/
```

#### 5. Strategy Testing and Conversion
```bash
# Test a strategy with sample data
uv run python cli.py strategies test simple --sample-file test.csv

# Test with overrides
uv run python cli.py strategies test customer_support \
  --sample-file tickets.csv \
  --overrides '{"embedder":{"config":{"batch_size":8}}}'

# Convert strategy to traditional config file
uv run python cli.py strategies convert simple simple_config.yaml
uv run python cli.py strategies convert customer_support support_config.yaml \
  --overrides '{"vector_store":{"config":{"collection_name":"custom_support"}}}'
```

### Document Ingestion (The Core Workflow)

#### 1. Basic Ingestion
```bash
# Single file ingestion (auto-detects file type)
uv run python cli.py --strategy simple ingest document.pdf
uv run python cli.py --strategy simple ingest data.csv
uv run python cli.py --strategy simple ingest report.docx

# Directory ingestion (processes all supported files)
uv run python cli.py --strategy simple ingest documents/
uv run python cli.py --strategy research ingest research_papers/
uv run python cli.py --strategy legal ingest legal_docs/
```

#### 2. Parser Override and File Type Control
```bash
# Override automatic parser detection
uv run python cli.py --strategy simple ingest data.txt --parser csv
uv run python cli.py --strategy simple ingest document.pdf --parser pdf_chunked

# Available parser overrides: csv, pdf, pdf_chunked, text, markdown, html
uv run python cli.py --strategy technical ingest docs.md --parser markdown
uv run python cli.py --strategy business ingest report.html --parser html
```

#### 3. Extractor Integration During Ingestion
```bash
# Apply extractors during ingestion
uv run python cli.py --strategy simple ingest documents.csv \
  --extractors yake entities statistics

# Configure extractors with custom settings
uv run python cli.py --strategy simple ingest documents.pdf \
  --extractors rake entities \
  --extractor-config '{"rake": {"max_keywords": 20}, "entities": {"entity_types": ["PERSON", "ORG"]}}'

# Available extractors: yake, rake, tfidf, entities, datetime, statistics, 
# summary, table, link, heading, pattern
```

#### 4. Advanced Ingestion Examples
```bash
# Multi-format business data ingestion
uv run python cli.py --strategy business ingest financial_reports/ \
  --extractors table statistics summary

# Customer support with comprehensive extraction
uv run python cli.py --strategy customer_support ingest support_data.csv \
  --extractors entities pattern summary statistics

# Research papers with academic extractors
uv run python cli.py --strategy research ingest papers/ \
  --extractors entities statistics summary heading

# Legal documents with compliance extraction
uv run python cli.py --strategy legal ingest contracts/ \
  --extractors entities datetime pattern heading statistics
```

### Search & Retrieval

#### 1. Basic Search
```bash
# Simple search with strategy
uv run python cli.py --strategy simple search "password reset"
uv run python cli.py --strategy customer_support search "login issues"
uv run python cli.py --strategy research search "machine learning algorithms"

# Control number of results
uv run python cli.py --strategy simple search "network error" --top-k 10
uv run python cli.py --strategy legal search "contract terms" --top-k 3
```

#### 2. Advanced Search with Filters
```bash
# Search with metadata filters
uv run python cli.py --strategy customer_support search "authentication" \
  --filter '{"priority": "high"}'

uv run python cli.py --strategy business search "revenue growth" \
  --filter '{"department": "finance", "year": 2024}'

# Complex filters with operators
uv run python cli.py --strategy legal search "liability" \
  --filter '{"priority": {"$in": ["high", "urgent"]}, "status": {"$ne": "closed"}}'

# Date-based filtering
uv run python cli.py --strategy research search "AI ethics" \
  --filter '{"publication_date": {"$gte": "2023-01-01"}}'
```

#### 3. Cross-Collection Search
```bash
# Search specific collections (after running demos)
uv run python cli.py search "transformer architecture" --collection research_papers
uv run python cli.py search "password reset procedures" --collection customer_support
uv run python cli.py search "API authentication methods" --collection code_documentation
uv run python cli.py search "quarterly performance metrics" --collection business_reports
```

#### 4. Search Strategy Comparison
```bash
# Compare different strategies on same query
uv run python cli.py --strategy simple search "security vulnerability"
uv run python cli.py --strategy customer_support search "security vulnerability"  
uv run python cli.py --strategy legal search "security vulnerability"

# Test different retrieval configurations
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"retrieval_strategy":{"config":{"distance_metric":"euclidean"}}}}' \
  search "data breach"
```

### Component Management & Testing

#### 1. List Available Components
```bash
# List parsers
uv run python cli.py parsers list

# List extractors with details
uv run python cli.py extractors list --detailed

# List embedding models
uv run python cli.py embedders list

# List vector stores
uv run python cli.py stores list
```

#### 2. Test Individual Extractors
```bash
# Test extractor with sample text
uv run python cli.py extractors test --extractor yake
uv run python cli.py extractors test --extractor entities --text "Contact John Doe at john@company.com for project updates"

# Test extractor with file
uv run python cli.py extractors test --extractor statistics --file sample_document.txt
uv run python cli.py extractors test --extractor rake --file research_paper.pdf

# Test with custom configuration
uv run python cli.py extractors test --extractor yake --text "Machine learning and artificial intelligence applications" \
  --config '{"max_keywords": 10, "deduplication_threshold": 0.7}'
```

#### 3. System Testing
```bash
# Test basic system functionality
uv run python cli.py test

# Test with specific file
uv run python cli.py test --test-file samples/test_document.pdf

# Test end-to-end pipeline
uv run python cli.py test --test-file samples/support_tickets.csv
```

### Document Management

#### 1. Vector Store Information
```bash
# Show collection information
uv run python cli.py info

# Show specific vector store info
uv run python cli.py info --vector-store custom_collection

# Get detailed statistics
uv run python cli.py manage stats --detailed
```

#### 2. Document Operations
```bash
# Delete operations
uv run python cli.py manage delete --older-than 30  # Delete docs older than 30 days
uv run python cli.py manage delete --doc-ids "doc1" "doc2" "doc3"
uv run python cli.py manage delete --strategy soft --filter '{"priority": "low"}'

# Replace documents
uv run python cli.py manage replace new_version.pdf --target-doc-id "old_doc_123"
uv run python cli.py manage replace updated_data.csv --target-doc-id "data_456" --strategy versioning

# Cleanup operations
uv run python cli.py manage cleanup --duplicates
uv run python cli.py manage cleanup --old-versions 5  # Keep only 5 latest versions
uv run python cli.py manage cleanup --expired
```

#### 3. Hash and Integrity Operations
```bash
# Find duplicate documents
uv run python cli.py manage hash --find-duplicates

# Verify document integrity
uv run python cli.py manage hash --verify-integrity

# Rehash documents (after content changes)
uv run python cli.py manage hash --rehash
```

### Advanced Usage Patterns

#### 1. Batch Processing Workflows
```bash
# Process multiple directories with different strategies
for dir in legal_docs customer_support research_papers; do
  strategy=$(case $dir in
    legal_docs) echo "legal" ;;
    customer_support) echo "customer_support" ;;
    research_papers) echo "research" ;;
  esac)
  uv run python cli.py --strategy $strategy ingest $dir/
done

# Bulk search across multiple collections
for query in "security policy" "data protection" "compliance requirements"; do
  echo "=== Searching: $query ==="
  uv run python cli.py --strategy legal search "$query" --top-k 3
done
```

#### 2. Environment-Specific Configurations
```bash
# Development environment
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"vector_store":{"config":{"persist_directory":"./dev_db"}}}}' \
  ingest test_data/

# Staging environment  
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"vector_store":{"config":{"persist_directory":"./staging_db"}}}}' \
  ingest staging_data/

# Production environment
uv run python cli.py --strategy production ingest production_data/
```

#### 3. Performance Optimization Examples
```bash
# High-performance ingestion (large datasets)
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":128}}}}' \
  ingest large_dataset/

# Memory-optimized processing (limited resources)
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
# Run all tests
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
└── 🧪 Tests: 118 tests with mocking and validation
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
        parser = YourParser({})
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
  type: "CSVParser"
  config:
    content_fields: ["title", "description"]
    metadata_fields: ["category", "priority"]

embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"
    batch_size: 16

vector_store:
  type: "ChromaStore"
  config:
    collection_name: "documents"
    persist_directory: "./vector_db"
```

```bash
# Use traditional config
uv run python cli.py --config config/custom.yaml ingest data.csv
```

## 🎯 Real-World Workflow Examples

### Complete Document Processing Workflows

#### Customer Support Knowledge Base Setup
```bash
# 1. Setup: Ingest support tickets with comprehensive extraction
uv run python cli.py --strategy customer_support ingest support_tickets.csv \
  --extractors entities pattern summary statistics

# 2. Search: Find similar cases
uv run python cli.py --strategy customer_support search "user cannot login" \
  --filter '{"priority": "high"}' --top-k 5

# 3. Management: Regular cleanup
uv run python cli.py manage cleanup --duplicates
uv run python cli.py manage delete --older-than 90
```

#### Legal Document Analysis Workflow
```bash
# 1. Ingest: Process legal documents with compliance extraction
uv run python cli.py --strategy legal ingest contracts/ \
  --extractors entities datetime pattern heading statistics

# 2. Research: Find relevant cases and precedents
uv run python cli.py --strategy legal search "liability clauses" \
  --filter '{"document_type": "contract", "jurisdiction": "US"}' --top-k 10

# 3. Analysis: Cross-reference different document types
uv run python cli.py --strategy legal search "force majeure" \
  --filter '{"date_range": {"$gte": "2020-01-01"}}' --top-k 15
```

#### Research Paper Processing Pipeline
```bash
# 1. Batch Processing: Ingest academic papers
uv run python cli.py --strategy research ingest research_papers/ \
  --extractors entities statistics summary heading

# 2. Literature Review: Find related work
uv run python cli.py --strategy research search "transformer architecture" \
  --filter '{"publication_year": {"$gte": 2020}}' --top-k 20

# 3. Citation Analysis: Track key researchers and institutions
uv run python cli.py --strategy research search "attention mechanisms" \
  --filter '{"authors": {"$contains": "Vaswani"}}' --top-k 10
```

#### Business Intelligence Workflow
```bash
# 1. Multi-format Ingestion: Process various business documents
uv run python cli.py --strategy business ingest quarterly_reports/ \
  --extractors table statistics summary

uv run python cli.py --strategy business ingest financial_data.xlsx \
  --extractors table statistics pattern

# 2. Strategic Analysis: Find performance trends
uv run python cli.py --strategy business search "revenue growth" \
  --filter '{"quarter": "Q4", "year": 2024}' --top-k 8

# 3. Comparative Analysis: Cross-department insights
uv run python cli.py --strategy business search "cost optimization" \
  --filter '{"department": {"$in": ["finance", "operations"]}}' --top-k 12
```

### Production Deployment Patterns

#### Development to Production Pipeline
```bash
# Development: Test with small dataset
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"vector_store":{"config":{"persist_directory":"./dev_db"}}}}' \
  ingest test_data/

# Staging: Validate with production-like data
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"vector_store":{"config":{"persist_directory":"./staging_db"}}}}' \
  ingest staging_data/

# Production: Full-scale deployment
uv run python cli.py --strategy production ingest production_data/
```

#### Performance Scaling Examples
```bash
# Small Dataset (< 1000 docs): Basic strategy
uv run python cli.py --strategy simple ingest small_docs/

# Medium Dataset (1000-10000 docs): Optimized batch processing
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":32}}}}' \
  ingest medium_docs/

# Large Dataset (> 10000 docs): High-performance configuration
uv run python cli.py --strategy production \
  --strategy-overrides '{
    "components": {
      "embedder": {"config": {"batch_size": 128}},
      "vector_store": {"config": {"batch_size": 1000}}
    }
  }' ingest large_docs/
```

## 🔍 Programmatic API

```python
from api import SearchAPI, search

# Quick search
results = search("password reset", top_k=3)
for result in results:
    print(f"Score: {result.score:.3f} - {result.content[:100]}...")

# Advanced usage
api = SearchAPI(config_path="config/custom.yaml")
results = api.search(
    query="security issue",
    top_k=5,
    metadata_filter={"priority": "high"}
)
```

## 🚨 Troubleshooting & Common Issues

### CLI Issues

#### Command Not Found or Import Errors
```bash
# Ensure dependencies are installed
uv sync

# Check Python environment
uv run python -c "import sys; print(sys.path)"

# Verify CLI works
uv run python cli.py --help

# Alternative: activate environment manually
source .venv/bin/activate
python cli.py --help
```

#### Strategy Configuration Problems
```bash
# List available strategies
uv run python cli.py strategies list

# Validate strategy syntax
uv run python cli.py strategies test simple

# Test override syntax
uv run python cli.py strategies test customer_support \
  --overrides '{"components":{"embedder":{"config":{"batch_size":8}}}}'

# Convert to traditional config for debugging
uv run python cli.py strategies convert simple debug_config.yaml
uv run python cli.py --config debug_config.yaml test
```

### Ollama Integration Issues

#### Ollama Service Problems
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Check available models
ollama list

# Pull required embedding model
ollama pull nomic-embed-text

# Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test embedding"
}'
```

#### Embedding Generation Failures
```bash
# Test with smaller batch size
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":1}}}}' \
  ingest small_test.csv

# Enable debug logging
uv run python cli.py --log-level DEBUG --strategy simple ingest test.csv

# Check available memory
free -h  # Linux
vm_stat  # macOS
```

### Performance Issues

#### Slow Ingestion
```bash
# Check current configuration
uv run python cli.py strategies show simple

# Optimize for speed
uv run python cli.py --strategy simple \
  --strategy-overrides '{
    "components": {
      "embedder": {"config": {"batch_size": 32}},
      "extractors": []
    }
  }' ingest large_dataset/

# Use production strategy for large datasets
uv run python cli.py --strategy production ingest large_dataset/
```

#### Memory Issues
```bash
# Reduce batch size
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":4}}}}' \
  ingest documents/

# Process files individually instead of directories
for file in documents/*.pdf; do
  uv run python cli.py --strategy simple ingest "$file"
done

# Check system resources during processing
top  # Monitor CPU/memory usage
```

### Search and Retrieval Issues

#### No Search Results
```bash
# Check collection info
uv run python cli.py info

# Verify documents were ingested
uv run python cli.py manage stats --detailed

# Test with broader search
uv run python cli.py --strategy simple search "common words" --top-k 10

# Check with different strategy
uv run python cli.py --strategy customer_support search "your query"
```

#### Poor Search Quality
```bash
# Try different retrieval strategies
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"retrieval_strategy":{"type":"MetadataFilteredStrategy"}}}' \
  search "your query"

# Increase number of results
uv run python cli.py --strategy simple search "query" --top-k 20

# Use more sophisticated strategy
uv run python cli.py --strategy production search "query"
```

### File Processing Issues

#### Unsupported File Types
```bash
# Check supported parsers
uv run python cli.py parsers list

# Override parser selection
uv run python cli.py --strategy simple ingest unknown_file.ext --parser text

# Convert file format if needed
pandoc document.doc -o document.pdf  # Convert to supported format
```

#### Parser Failures
```bash
# Test file parsing separately
uv run python cli.py test --test-file problematic_document.pdf

# Try different parser
uv run python cli.py --strategy simple ingest document.pdf --parser pdf_chunked

# Enable debug logging
uv run python cli.py --log-level DEBUG --strategy simple ingest document.pdf
```

### Database and Storage Issues

#### ChromaDB Connection Problems
```bash
# Check database directory permissions
ls -la ./data/simple_chroma_db/

# Create directory manually if needed
mkdir -p ./data/simple_chroma_db

# Use custom database location
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"vector_store":{"config":{"persist_directory":"./custom_db"}}}}' \
  ingest documents/
```

#### Collection Issues
```bash
# List collection info
uv run python cli.py info

# Clear and recreate collection
uv run python cli.py manage cleanup --duplicates

# Use different collection name
uv run python cli.py --strategy simple \
  --strategy-overrides '{"components":{"vector_store":{"config":{"collection_name":"new_collection"}}}}' \
  ingest documents/
```

### Performance Optimization Guide

#### Hardware-Based Recommendations

**Low Resources (< 8GB RAM, CPU only):**
```bash
# Use minimal strategy with small batches
uv run python cli.py --strategy simple \
  --strategy-overrides '{
    "components": {
      "embedder": {"config": {"batch_size": 2}},
      "extractors": []
    }
  }' ingest documents/
```

**Medium Resources (8-16GB RAM, integrated GPU):**
```bash
# Balanced configuration
uv run python cli.py --strategy customer_support \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":16}}}}' \
  ingest documents/
```

**High Resources (16GB+ RAM, dedicated GPU):**
```bash
# High-performance configuration
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":128}}}}' \
  ingest documents/
```

#### Dataset Size Recommendations

**Small (< 1,000 documents):**
```bash
uv run python cli.py --strategy simple ingest small_docs/
```

**Medium (1,000 - 50,000 documents):**
```bash
uv run python cli.py --strategy production \
  --strategy-overrides '{"components":{"embedder":{"config":{"batch_size":32}}}}' \
  ingest medium_docs/
```

**Large (50,000+ documents):**
```bash
# Process in chunks
split -l 10000 large_dataset.csv chunk_
for chunk in chunk_*; do
  uv run python cli.py --strategy production ingest "$chunk"
done
```

### Quick Reference Commands

#### Essential Daily Commands
```bash
# Quick start
uv run python cli.py --strategy simple ingest documents/
uv run python cli.py --strategy simple search "query"

# System check
uv run python cli.py strategies list
uv run python cli.py info
uv run python cli.py test

# Maintenance
uv run python cli.py manage cleanup --duplicates
uv run python cli.py manage stats --detailed
```

#### Emergency Recovery
```bash
# Reset database
rm -rf ./data/simple_chroma_db/
uv run python cli.py --strategy simple ingest documents/

# Test system health
uv run python -m pytest tests/ -v

# Verify Ollama
curl http://localhost:11434/api/tags
```

## 📚 Documentation

- **[STRUCTURE.md](STRUCTURE.md)**: Complete developer architecture guide
- **[demos/README.md](demos/README.md)**: Detailed demo documentation
- **[schema.yaml](schema.yaml)**: Complete API documentation
- **[METADATA_BEST_PRACTICES.md](METADATA_BEST_PRACTICES.md)**: Metadata management guide

## 🤝 Contributing

1. **Follow the patterns**: Use factory registration, base classes, and configuration schemas
2. **Write tests**: All new components need comprehensive tests
3. **Update documentation**: Keep STRUCTURE.md and schema.yaml current
4. **Test thoroughly**: Run the full test suite and demos
5. **Add examples**: Show how to use new components

## 📈 What's Next?

The RAG system is designed for continuous extension:

### Immediate Next Steps
- Add more vector stores (Pinecone, Qdrant, Weaviate)
- Expand parser support (Word, XML, JSON, Code)
- Add more embedding providers (OpenAI, Cohere)
- Build web interface and REST API

### Architecture Evolution
- Plugin system for third-party extensions
- Distributed processing capabilities
- Advanced metadata management
- Performance monitoring and analytics

The modular architecture ensures all extensions integrate seamlessly with existing components and strategies.

---

**Get started in under 2 minutes:**
```bash
cd rag/
uv sync
ollama serve &
ollama pull nomic-embed-text
uv run python demos/enhanced_demo.py
```

**Questions?** Check out the [demos](demos/), run the [tests](tests/), or dive into the [architecture guide](STRUCTURE.md).