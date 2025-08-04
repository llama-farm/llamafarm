# Simple Extensible RAG System

A lightweight, extensible RAG (Retrieval-Augmented Generation) system designed for easy contribution and customization. Built with a modular architecture that makes it simple to add new data sources, embedding models, and vector databases.

## Features

- **Extensible Architecture**: Easy to add new parsers, embedders, and vector stores
- **CSV Support**: Built-in support for customer support tickets and other tabular data
- **Ollama Integration**: Local embeddings using Ollama
- **ChromaDB Support**: Persistent vector storage
- **Simple CLI**: Easy setup and management
- **Configuration-Driven**: JSON configuration files
- **Testing Framework**: Comprehensive tests included

## Quick Start

### 🚀 Automated Setup (Recommended)

**For macOS users**, we provide automated setup scripts:

```bash
# Full setup and demo (recommended for first-time users)
./setup_and_demo.sh

# Quick extractor testing only (no full system setup)
./scripts/quick_extractor_demo.sh

# Automated setup without prompts
./setup_and_demo.sh --skip-prompts

# Run tests only
./setup_and_demo.sh --tests-only
```

The setup script will:
- ✅ Install dependencies (uv, Ollama, Python packages)
- ✅ Set up virtual environment
- ✅ Download embedding models
- ✅ Run comprehensive demos of all features
- ✅ Show usage examples

### 📋 Manual Setup

If you prefer manual setup or are not on macOS:

#### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for embeddings)

#### macOS Installation with UV (Recommended)
#### macOS Installation with UV (Recommended)

1. **Install UV (the fast Python package manager)**:
   ```bash
   # Method 1: Official installer
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Method 2: Homebrew
   brew install uv
   
   # Method 3: pipx
   pipx install uv
   ```

2. **Install Ollama**:
   ```bash
   # Method 1: Official installer
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Method 2: Homebrew (alternative)
   brew install ollama
   ```

3. **Start Ollama and pull the embedding model**:
   ```bash
   # Start Ollama service
   ollama serve
   
   # In a new terminal, pull the embedding model
   ollama pull nomic-embed-text
   ```

4. **Setup the project with UV**:
   ```bash
   # Clone the repository
   cd rag/
   
   # Create virtual environment and install dependencies
   uv sync
   
   # Activate the environment  
   source .venv/bin/activate
   
   # Or run commands directly with uv
   uv run python cli.py --help
   ```

### Alternative Installation (pip/venv)

If you prefer traditional pip/venv:
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   
   # Install with pip
   pip install -e .
   ```

### Basic Usage

1. **Test the system:**
1. **Test the system:**
   ```bash
   # Test CSV parsing
   uv run python cli.py test --test-file samples/small_sample.csv 
   
   # Test PDF parsing (if you have a PDF file)
   uv run python cli.py test --test-file samples/test_document.pdf
   ```

## 🔧 Local-Only Extractors

The RAG system includes 5 local-only extractors that enhance documents with metadata **without requiring external LLMs**:

### Available Extractors

- **YAKE**: Advanced keyword extraction considering position and context
- **RAKE**: Fast phrase extraction using stop words as delimiters  
- **TF-IDF**: Term frequency analysis for finding unique terms
- **Entities**: Person, organization, date, email, phone extraction (spaCy + regex fallbacks)
- **DateTime**: Date, time, and relative date extraction
- **Statistics**: Readability metrics, vocabulary analysis, content statistics

### Extractor Commands

```bash
# List all available extractors
uv run python cli.py extractors list --detailed

# Test an extractor on sample text
uv run python cli.py extractors test --extractor yake

# Test with your own text
uv run python cli.py extractors test --extractor entities --text "Contact John Doe at john@company.com"

# Test with a file
uv run python cli.py extractors test --extractor statistics --file samples/document.txt
```

### Using Extractors During Ingestion

```bash
# Apply extractors during CSV ingestion
uv run python cli.py ingest samples/small_sample.csv --extractors yake entities statistics

# Apply with custom configuration
uv run python cli.py ingest document.pdf --extractors rake entities \
  --extractor-config '{"rake": {"max_keywords": 20}, "entities": {"entity_types": ["PERSON", "ORG"]}}'

# Use configuration-based extractors (see config_examples/extractors_demo_config.yaml)
uv run python cli.py --config config_examples/extractors_demo_config.yaml ingest samples/document.pdf
```

### Extractor Output Example

```json
{
  "yake_keywords": ["machine learning", "artificial intelligence", "data analysis"],
  "entities_person": ["John Smith", "Sarah Johnson"],
  "entities_email": ["contact@company.com"],
  "word_count": 1250,
  "reading_time_minutes": 6.25,
  "sentiment_classification": "positive",
  "flesch_reading_ease": 65.2
}
```

## 📊 CSV Processing

### CSV Commands with Sample Data

```bash
# 1. Test CSV parsing with sample data
uv run python cli.py --config config_examples/basic_config.yaml \
  test --test-file samples/small_sample.csv

# 2. Ingest CSV data using specific configuration
uv run python cli.py --config config_examples/basic_config.yaml \
  ingest samples/small_sample.csv

# 3. Search the ingested CSV data
uv run python cli.py --config config_examples/basic_config.yaml \
  search "login problems"

# 4. Check collection info
uv run python cli.py --config config_examples/basic_config.yaml info
```

### Custom CSV Configuration

```bash
# Use custom CSV configuration for different CSV formats
uv run python cli.py --config config_examples/custom_csv_config.yaml \
  ingest your_custom_data.csv
```

## 📄 PDF Processing

### Single PDF Document

```bash
# 1. Test PDF parsing with sample document
uv run python cli.py --config config_examples/pdf_config.yaml \
  test --test-file samples/test_document.pdf

# 2. Ingest single PDF (combined pages mode)
uv run python cli.py --config config_examples/pdf_config.yaml \
  ingest samples/test_document.pdf

# 3. Search PDF content
uv run python cli.py --config config_examples/pdf_config.yaml \
  search "specific topic from PDF"
```

### Multiple PDF Files

```bash
# Process multiple PDFs by running command for each file
uv run python cli.py --config config_examples/pdf_config.yaml \
  ingest samples/document1.pdf

uv run python cli.py --config config_examples/pdf_config.yaml \
  ingest samples/document2.pdf

uv run python cli.py --config config_examples/pdf_config.yaml \
  ingest samples/document3.pdf
```

### PDF Directory Processing

```bash
# Process all PDFs in a directory using a script
# First, create a simple script to process multiple files:
for pdf in samples/pdfs/*.pdf; do
  echo "Processing: $pdf"
  uv run python cli.py --config config_examples/pdf_config.yaml ingest "$pdf"
done

# Or use find to process PDFs recursively
find samples/ -name "*.pdf" -exec uv run python cli.py \
  --config config_examples/pdf_config.yaml ingest {} \;
```

### PDF Page-by-Page Processing

```bash
# Use separate pages configuration for page-level search
uv run python cli.py --config config_examples/pdf_separate_pages_config.yaml \
  ingest samples/multi_page_document.pdf

# Search will return individual pages as separate results
uv run python cli.py --config config_examples/pdf_separate_pages_config.yaml \
  search "chapter introduction"
```

### Advanced PDF Processing Examples

```bash
# Process PDFs with custom base directory
uv run python cli.py --base-dir /path/to/project \
  --config config_examples/pdf_config.yaml \
  ingest data/documents/report.pdf

# Process from different directory
cd /different/directory
uv run python /path/to/rag/cli.py \
  --config /path/to/rag/config_examples/pdf_config.yaml \
  ingest ~/Documents/important_document.pdf
```

## 🔍 Search Examples

```bash
# Search CSV data with basic retrieval strategy
uv run python cli.py --config config_examples/basic_with_retrieval_config.yaml \
  search "password reset" --top-k 3

# Search with advanced hybrid retrieval strategy
uv run python cli.py --config config_examples/advanced_retrieval_config.yaml \
  search "login authentication" --top-k 5

# Search PDF data with metadata filtering
uv run python cli.py --config config_examples/pdf_with_retrieval_config.json \
  search "methodology" --top-k 5

# Compare different retrieval strategies
uv run python cli.py --config config_examples/basic_with_retrieval_config.yaml search "security issue"
uv run python cli.py --config config_examples/advanced_retrieval_config.yaml search "security issue"
```

### 🧪 Testing and Examples

```bash
# Test the entire retrieval system
uv run python examples/test_retrieval_system.py

# Explore retrieval strategies with examples
uv run python examples/example_retrieval_usage.py

# Basic API usage examples
uv run python examples/example_api_usage.py
```

## 🛤️ Flexible Path Resolution

The RAG system supports flexible path resolution for both configuration files and data sources, making it easy to work with different project structures and deployment scenarios.

### Configuration File Paths

Config files are searched in multiple locations (in order of preference):

```bash
# 1. Relative to current/base directory
uv run python cli.py --config my_config.json

# 2. Relative to user config directory
uv run python cli.py --config config.json  # Searches ~/.config/rag/config.json

# 3. Absolute path
uv run python cli.py --config /etc/rag/production.json

# 4. From subdirectory
uv run python cli.py --config config_examples/flexible_paths_config.json
```

### Data Source Paths

Data sources support various path formats:

```bash
# Relative paths (resolved from current or base directory)
uv run python cli.py ingest samples/small_sample.csv
uv run python cli.py ingest data/tickets.csv

# Absolute paths
uv run python cli.py ingest /path/to/data/tickets.csv

# Home directory expansion
uv run python cli.py ingest ~/Documents/support_data.csv

# With custom base directory
uv run python cli.py --base-dir /project/root ingest data/tickets.csv
```

### Base Directory Override

Use `--base-dir` to change the reference point for relative paths:

```bash
# Set custom base directory for all relative paths
uv run python cli.py --base-dir /my/project/root \
  --config configs/prod.json \
  ingest data/latest_tickets.csv
```

### Path Resolution in Config Files

Paths within configuration files are automatically resolved:

```json
{
  "vector_store": {
    "type": "ChromaStore",
    "config": {
      "persist_directory": "./data/chroma_db"  // Auto-resolved and created
    }
  }
}
```

### Alternative: Run the Example Script

```bash
# With UV
uv run python examples/example.py

# Or with activated environment
python examples/example.py
```

### Development Commands

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Install development dependencies
uv sync --dev

# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name
```

## Architecture

The system is built around four core components:

### 1. Parsers
Convert various data formats into the universal `Document` format.

- `CSVParser`: Generic CSV parser with configurable fields
- `CustomerSupportCSVParser`: Specialized for support ticket data
- `PDFParser`: Extract text, metadata, and structure from PDF documents
- `PDFParser`: Extract text, metadata, and structure from PDF documents

### 2. Embedders
Generate vector embeddings from text content.

- `OllamaEmbedder`: Uses local Ollama models

### 3. Vector Stores
Store and search document embeddings.

- `ChromaStore`: ChromaDB integration with persistence

### 4. Universal Retrieval Strategies
Advanced, database-agnostic retrieval strategies that automatically optimize for your vector database.

- `BasicSimilarityStrategy`: Simple vector similarity search (great for getting started)
- `MetadataFilteredStrategy`: Intelligent metadata filtering with native/fallback support
- `MultiQueryStrategy`: Uses multiple query variations to improve recall  
- `RerankedStrategy`: Multi-factor re-ranking for sophisticated relevance scoring
- `HybridUniversalStrategy`: Combines multiple strategies with configurable weights

### 5. Pipeline
### 4. Universal Retrieval Strategies
Advanced, database-agnostic retrieval strategies that automatically optimize for your vector database.

- `BasicSimilarityStrategy`: Simple vector similarity search (great for getting started)
- `MetadataFilteredStrategy`: Intelligent metadata filtering with native/fallback support
- `MultiQueryStrategy`: Uses multiple query variations to improve recall  
- `RerankedStrategy`: Multi-factor re-ranking for sophisticated relevance scoring
- `HybridUniversalStrategy`: Combines multiple strategies with configurable weights

### 5. Pipeline
Chains components together for end-to-end processing.

## 🎯 Universal Retrieval Strategies

The system features a powerful **database-agnostic retrieval system** that automatically optimizes strategies based on your vector database capabilities. All strategies work with any vector database while automatically using database-specific optimizations when available.

### Available Strategies

#### **BasicSimilarityStrategy** - Getting Started
```json
{
  "retrieval_strategy": {
    "type": "BasicSimilarityStrategy",
    "config": {
      "distance_metric": "cosine"
    }
  }
}
```
- **Use Cases**: Getting started, simple semantic search, baseline testing
- **Performance**: Fast | **Complexity**: Low

#### **MetadataFilteredStrategy** - Smart Filtering  
```json
{
  "retrieval_strategy": {
    "type": "MetadataFilteredStrategy",
    "config": {
      "distance_metric": "cosine",
      "default_filters": {
        "priority": ["high", "medium"],
        "type": "documentation"
      },
      "fallback_multiplier": 3
    }
  }
}
```
- **Features**: Native filtering when supported, automatic fallback, complex operators (`$ne`, `$in`, `$gt`, etc.)
- **Use Cases**: Domain-specific searches, multi-tenant applications, content categorization
- **Performance**: Medium | **Complexity**: Medium

#### **MultiQueryStrategy** - Enhanced Recall
```json
{
  "retrieval_strategy": {
    "type": "MultiQueryStrategy", 
    "config": {
      "num_queries": 3,
      "aggregation_method": "weighted",
      "search_multiplier": 2
    }
  }
}
```
- **Aggregation Methods**: `max` (best score), `mean` (average), `weighted` (decreasing weights)
- **Use Cases**: Ambiguous queries, query expansion, improving recall for complex questions
- **Performance**: Medium | **Complexity**: Medium

#### **RerankedStrategy** - Sophisticated Ranking
```json
{
  "retrieval_strategy": {
    "type": "RerankedStrategy",
    "config": {
      "initial_k": 20,
      "length_normalization": 1000,
      "rerank_factors": {
        "recency": 0.1,
        "length": 0.05, 
        "metadata_boost": 0.2
      }
    }
  }
}
```
- **Reranking Factors**: Recency boost, content length preference, metadata-based boosts (priority, type, quality indicators)
- **Use Cases**: Production systems, time-sensitive content, multi-factor relevance
- **Performance**: Slower | **Complexity**: High

#### **HybridUniversalStrategy** - Best of All Worlds
```json
{
  "retrieval_strategy": {
    "type": "HybridUniversalStrategy",
    "config": {
      "combination_method": "weighted_average",
      "normalize_scores": true,
      "diversity_boost": 0.1,
      "strategies": [
        {"type": "BasicSimilarityStrategy", "weight": 0.4},
        {"type": "MetadataFilteredStrategy", "weight": 0.3},
        {"type": "RerankedStrategy", "weight": 0.2},
        {"type": "MultiQueryStrategy", "weight": 0.1}
      ]
    }
  }
}
```
- **Combination Methods**: `weighted_average` (score-based), `rank_fusion` (position-based)
- **Features**: Strategy aliases (`basic`, `filtered`, `multi_query`, `reranked`)
- **Use Cases**: Production systems, balanced precision/recall, complex requirements  
- **Performance**: Variable | **Complexity**: High

### Strategy Selection Guide

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| **Getting Started** | `BasicSimilarityStrategy` | Simple, fast, reliable baseline |
| **Production General** | `HybridUniversalStrategy` | Balanced performance across use cases |
| **High Precision** | `RerankedStrategy` | Multi-factor ranking for nuanced results |
| **Filtered Content** | `MetadataFilteredStrategy` | Efficient domain-specific searches |
| **Complex Queries** | `MultiQueryStrategy` | Better recall for ambiguous questions |
| **High Performance** | `BasicSimilarityStrategy` | Minimal overhead, maximum speed |

## 🎯 Universal Retrieval Strategies

The system features a powerful **database-agnostic retrieval system** that automatically optimizes strategies based on your vector database capabilities. All strategies work with any vector database while automatically using database-specific optimizations when available.

### Available Strategies

#### **BasicSimilarityStrategy** - Getting Started
```json
{
  "retrieval_strategy": {
    "type": "BasicSimilarityStrategy",
    "config": {
      "distance_metric": "cosine"
    }
  }
}
```
- **Use Cases**: Getting started, simple semantic search, baseline testing
- **Performance**: Fast | **Complexity**: Low

#### **MetadataFilteredStrategy** - Smart Filtering  
```json
{
  "retrieval_strategy": {
    "type": "MetadataFilteredStrategy",
    "config": {
      "distance_metric": "cosine",
      "default_filters": {
        "priority": ["high", "medium"],
        "type": "documentation"
      },
      "fallback_multiplier": 3
    }
  }
}
```
- **Features**: Native filtering when supported, automatic fallback, complex operators (`$ne`, `$in`, `$gt`, etc.)
- **Use Cases**: Domain-specific searches, multi-tenant applications, content categorization
- **Performance**: Medium | **Complexity**: Medium

#### **MultiQueryStrategy** - Enhanced Recall
```json
{
  "retrieval_strategy": {
    "type": "MultiQueryStrategy", 
    "config": {
      "num_queries": 3,
      "aggregation_method": "weighted",
      "search_multiplier": 2
    }
  }
}
```
- **Aggregation Methods**: `max` (best score), `mean` (average), `weighted` (decreasing weights)
- **Use Cases**: Ambiguous queries, query expansion, improving recall for complex questions
- **Performance**: Medium | **Complexity**: Medium

#### **RerankedStrategy** - Sophisticated Ranking
```json
{
  "retrieval_strategy": {
    "type": "RerankedStrategy",
    "config": {
      "initial_k": 20,
      "length_normalization": 1000,
      "rerank_factors": {
        "recency": 0.1,
        "length": 0.05, 
        "metadata_boost": 0.2
      }
    }
  }
}
```
- **Reranking Factors**: Recency boost, content length preference, metadata-based boosts (priority, type, quality indicators)
- **Use Cases**: Production systems, time-sensitive content, multi-factor relevance
- **Performance**: Slower | **Complexity**: High

#### **HybridUniversalStrategy** - Best of All Worlds
```json
{
  "retrieval_strategy": {
    "type": "HybridUniversalStrategy",
    "config": {
      "combination_method": "weighted_average",
      "normalize_scores": true,
      "diversity_boost": 0.1,
      "strategies": [
        {"type": "BasicSimilarityStrategy", "weight": 0.4},
        {"type": "MetadataFilteredStrategy", "weight": 0.3},
        {"type": "RerankedStrategy", "weight": 0.2},
        {"type": "MultiQueryStrategy", "weight": 0.1}
      ]
    }
  }
}
```
- **Combination Methods**: `weighted_average` (score-based), `rank_fusion` (position-based)
- **Features**: Strategy aliases (`basic`, `filtered`, `multi_query`, `reranked`)
- **Use Cases**: Production systems, balanced precision/recall, complex requirements  
- **Performance**: Variable | **Complexity**: High

### Strategy Selection Guide

| Use Case | Recommended Strategy | Why |
|----------|---------------------|-----|
| **Getting Started** | `BasicSimilarityStrategy` | Simple, fast, reliable baseline |
| **Production General** | `HybridUniversalStrategy` | Balanced performance across use cases |
| **High Precision** | `RerankedStrategy` | Multi-factor ranking for nuanced results |
| **Filtered Content** | `MetadataFilteredStrategy` | Efficient domain-specific searches |
| **Complex Queries** | `MultiQueryStrategy` | Better recall for ambiguous questions |
| **High Performance** | `BasicSimilarityStrategy` | Minimal overhead, maximum speed |

## Configuration

Configuration is JSON-based and includes three main sections. The system supports flexible configuration file placement and automatic path resolution.

### CLI Configuration Options

```bash
# Global options (available for all commands)
--config, -c     Configuration file path (default: rag_config.json)
--base-dir, -b   Base directory for relative path resolution
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Configuration File Structure
Configuration is JSON-based and includes three main sections. The system supports flexible configuration file placement and automatic path resolution.

### CLI Configuration Options

```bash
# Global options (available for all commands)
--config, -c     Configuration file path (default: rag_config.json)
--base-dir, -b   Base directory for relative path resolution
--log-level      Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Configuration File Structure

```json
{
  "parser": {
    "type": "CustomerSupportCSVParser",
    "config": {
      "content_fields": ["subject", "body", "answer"],
      "metadata_fields": ["type", "priority", "language"]
    }
  },
  "embedder": {
    "type": "OllamaEmbedder",
    "config": {
      "model": "nomic-embed-text",
      "batch_size": 16
    }
  },
  "vector_store": {
    "type": "ChromaStore",
    "config": {
      "collection_name": "support_tickets",
      "persist_directory": "./chroma_db"
    }
  }
}
```

### Sample Configurations

#### Basic Configurations
- `config_examples/basic_config.yaml`: Simple setup without retrieval strategies
- `config_examples/custom_csv_config.yaml`: Custom CSV format configuration
- `config_examples/flexible_paths_config.json`: Demonstrates flexible path resolution

#### PDF Processing
- `config_examples/pdf_config.yaml`: PDF processing with combined pages
- `config_examples/pdf_separate_pages_config.yaml`: PDF processing with separate page documents  
- `config_examples/pdf_with_retrieval_config.json`: PDF processing with retrieval strategies

#### Retrieval Strategy Examples
- `config_examples/universal_retrieval_config.json`: Basic similarity strategy
- `config_examples/metadata_filtered_config.json`: Smart metadata filtering
- `config_examples/multi_query_retrieval_config.json`: Enhanced recall with multiple queries
- `config_examples/reranked_strategy_config.json`: Sophisticated multi-factor ranking
- `config_examples/hybrid_universal_config.json`: Weighted strategy combination
- `config_examples/rank_fusion_hybrid_config.json`: Rank fusion combination method
- `config_examples/advanced_retrieval_config.yaml`: Complex 4-strategy hybrid
- `config_examples/strategy_aliases_config.json`: Using strategy aliases (`basic`, `filtered`, etc.)
- `config_examples/production_config.json`: Production-ready configuration

## Adding New Components

### Adding a New Parser

1. **Create a new parser class:**

```python
from core.base import Parser, Document, ProcessingResult

class MyCustomParser(Parser):
    def parse(self, source: str) -> ProcessingResult:
        documents = []
        # Your parsing logic here
        return ProcessingResult(documents=documents)
```

2. **Register in CLI:**

Add to `create_pipeline_from_config()` in `cli.py`:

```python
elif parser_type == "MyCustomParser":
    parser = MyCustomParser(config=parser_config.get("config", {}))
```

### Adding a New Embedder

1. **Create embedder class:**

```python
from core.base import Embedder

class MyEmbedder(Embedder):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Your embedding logic here
        return embeddings
```

2. **Register in CLI and update configuration**

### Adding a New Vector Store

1. **Create store class:**

```python  
from core.base import VectorStore, Document

class MyVectorStore(VectorStore):
    def add_documents(self, documents: List[Document]) -> bool:
        # Storage logic
        pass
    
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        # Search logic
        pass
```

2. **Register in CLI**

## Examples

### Programmatic Usage

```python
from core.base import Pipeline
from parsers.csv_parser import CustomerSupportCSVParser
from embedders.ollama_embedder import OllamaEmbedder
from stores.chroma_store import ChromaStore

# Create pipeline
pipeline = Pipeline("My RAG Pipeline")
pipeline.add_component(CustomerSupportCSVParser())
pipeline.add_component(OllamaEmbedder())
pipeline.add_component(ChromaStore())

# Process data
result = pipeline.run(source="my_data.csv")
```

### 🔍 Using the Internal Search API

The RAG system includes an internal Python API for programmatic search access:

```python
from api import SearchAPI, search

# Quick search using convenience function
results = search("password reset", top_k=3)
for result in results:
    print(f"Score: {result.score:.3f} - {result.content[:100]}...")

# Advanced usage with API class
api = SearchAPI(config_path="config_examples/basic_config.yaml")

# Search with filters
results = api.search(
    query="security issue",
    top_k=5,
    min_score=-500.0,  # Only high-confidence results
    metadata_filter={"priority": "high"}  # Only high priority items
)

# Get raw Document objects
documents = api.search("network error", return_raw_documents=True)

# Get collection info
info = api.get_collection_info()
```

#### API Features:
- **Configurable search parameters**: `top_k`, `min_score`, `metadata_filter`
- **Multiple return formats**: `SearchResult` objects or raw `Document` objects
- **Metadata filtering**: Filter by any metadata field (priority, tags, etc.)
- **Collection information**: Get stats about the vector store
- **Multiple configurations**: Use different configs for different data types

Run the example script to see all features:
```bash
uv run python examples/example_api_usage.py
```

### 🔍 Advanced Retrieval Strategies

The RAG system now supports configurable retrieval strategies for fine-tuned search behavior:

```python
from api import SearchAPI

# Use different retrieval strategies via configuration
basic_api = SearchAPI("config_examples/basic_with_retrieval_config.yaml")
advanced_api = SearchAPI("config_examples/advanced_retrieval_config.yaml")

# Basic strategy - simple vector similarity
basic_results = basic_api.search("password reset", top_k=3)

# Advanced hybrid strategy - combines multiple approaches  
advanced_results = advanced_api.search("login authentication", top_k=5)

# Compare strategy effectiveness
print(f"Basic strategy found {len(basic_results)} results")
print(f"Advanced strategy found {len(advanced_results)} results")
```

#### Available Retrieval Strategies:

1. **ChromaBasicStrategy** - Simple vector similarity search
   ```json
   {
     "retrieval_strategy": {
       "type": "ChromaBasicStrategy",
       "config": {"distance_metric": "cosine"}
     }
   }
   ```

2. **ChromaHybridStrategy** - Combines multiple strategies with weights
   ```json
   {
     "retrieval_strategy": {
       "type": "ChromaHybridStrategy", 
       "config": {
         "strategies": [
           {"type": "ChromaBasicStrategy", "weight": 0.5},
           {"type": "ChromaRerankedStrategy", "weight": 0.3},
           {"type": "ChromaMetadataFilterStrategy", "weight": 0.2}
         ]
       }
     }
   }
   ```

3. **ChromaMetadataFilterStrategy** - Vector search with metadata filtering
4. **ChromaMultiQueryStrategy** - Uses multiple query variations  
5. **ChromaRerankedStrategy** - Re-ranks results with multiple factors

#### Testing Retrieval Strategies:

```bash
# Test the retrieval system
uv run python test_retrieval_system.py

# Run retrieval examples  
uv run python example_retrieval_usage.py

# Compare strategies side-by-side
uv run python cli.py --config config_examples/basic_with_retrieval_config.yaml search "security issue"
uv run python cli.py --config config_examples/advanced_retrieval_config.yaml search "security issue"
```

See `retrieval/README.md` for detailed documentation on all available strategies and configuration options.

### Custom CSV Format

For custom CSV formats, configure field mappings:

```json
{
  "parser": {
    "type": "CSVParser",
    "config": {
      "content_fields": ["title", "description", "solution"],
      "metadata_fields": ["category", "severity", "date"],
      "id_field": "ticket_number",
      "combine_content": true,
      "content_separator": "\\n\\n"
    }
  }
}
```

## Testing

Run the comprehensive test suite:

```bash
python examples/test_system.py
```

This tests:
- CSV parsing functionality
- Ollama integration (if available)
- ChromaDB storage
- End-to-end pipeline

## File Structure

```
rag/
├── core/
│   ├── __init__.py
│   ├── base.py              # Base classes
│   ├── enhanced_pipeline.py # Enhanced pipeline with progress tracking  
│   └── factories.py         # Factory pattern for component creation
│   ├── base.py              # Base classes
│   ├── enhanced_pipeline.py # Enhanced pipeline with progress tracking  
│   └── factories.py         # Factory pattern for component creation
├── parsers/
│   ├── __init__.py
│   ├── csv_parser.py        # CSV parsers
│   └── pdf_parser.py        # PDF parser with PyPDF2
│   ├── csv_parser.py        # CSV parsers
│   └── pdf_parser.py        # PDF parser with PyPDF2
├── embedders/
│   ├── __init__.py
│   └── ollama_embedder.py   # Ollama integration
├── stores/
│   ├── __init__.py
│   └── chroma_store.py      # ChromaDB integration
├── utils/
│   ├── __init__.py
│   ├── progress.py          # Progress tracking with llama puns
│   └── path_resolver.py     # Flexible path resolution
├── tests/                   # Comprehensive test suite
│   ├── test_csv_parser.py
│   ├── test_pdf_parser.py
│   └── ...
├── utils/
│   ├── __init__.py
│   ├── progress.py          # Progress tracking with llama puns
│   └── path_resolver.py     # Flexible path resolution
├── tests/                   # Comprehensive test suite
│   ├── test_csv_parser.py
│   ├── test_pdf_parser.py
│   └── ...
├── config_examples/         # Sample configurations
│   ├── basic_config.json
│   ├── custom_csv_config.json
│   ├── pdf_config.json
│   └── pdf_separate_pages_config.json
├── samples/                 # Sample data files
│   └── small_sample.csv
│   ├── basic_config.json
│   ├── custom_csv_config.json
│   ├── pdf_config.json
│   └── pdf_separate_pages_config.json
├── samples/                 # Sample data files
│   └── small_sample.csv
├── cli.py                   # Command-line interface
├── pyproject.toml          # UV project configuration
├── pyproject.toml          # UV project configuration
└── README.md              # This file
```

## Contributing

This system is designed for easy extension. To contribute:

1. **Add new components** following the base class patterns
2. **Update CLI registration** for new component types
3. **Add tests** for new functionality
4. **Update documentation** with usage examples

### Component Guidelines

- Inherit from appropriate base classes (`Parser`, `Embedder`, `VectorStore`)
- Implement required abstract methods
- Include configuration validation
- Add error handling and logging
- Follow the established patterns

## Troubleshooting

### macOS-Specific Issues

#### Ollama Installation Problems
```bash
# If the curl install fails, try Homebrew
brew install ollama

# Start Ollama
ollama serve
```

#### Python/pip Issues
```bash
# Use Python 3 explicitly if needed
# Create your rag_config.json file first (see Configuration section)
python3 cli.py test

# Create virtual environment if you get permission errors
python3 -m venv rag_env
source rag_env/bin/activate
pip install -r requirements.txt
```

#### Check if Everything is Working
```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Run comprehensive system tests with UV
uv run python examples/test_system.py

# Or run pytest
uv run pytest
```

### General Issues

#### Ollama Problems
- Ensure Ollama is running: `ollama serve`
- Check model availability: `ollama list`
- Pull required model: `ollama pull nomic-embed-text`

#### ChromaDB Issues
- Check permissions on persist directory
- Ensure disk space available
- Try deleting collection and recreating

#### Performance Tips
- Adjust batch sizes based on your hardware
- Use appropriate embedding models for your use case
- Consider chunking large CSV files

## 🚀 Next Steps & Roadmap

The RAG system is designed for continuous extension and improvement. Here are the planned next steps for expanding capabilities:

### 🔧 Infrastructure Improvements

#### Top-Level Configuration Library Integration
- Replace current JSON config loading with centralized config management
- Support for environment-based configuration
- Configuration validation and schema enforcement
- Hot-reload capabilities for development

#### Global Logging Module Integration
- Replace current basic logging with enterprise-grade logging system
- Structured logging with JSON output
- Log aggregation and monitoring integration
- Configurable log levels per component

#### Advanced Component Registration
- Dynamic component discovery and registration
- Plugin system for third-party extensions
- Runtime component loading from external packages
- Component dependency management

### 📄 New Parser Support

#### Document Format Expansion
- ✅ **PDF Parser**: Extract text, metadata, and structure from PDF documents (COMPLETED)
- ✅ **PDF Parser**: Extract text, metadata, and structure from PDF documents (COMPLETED)
- **Word Document Parser**: Support for .docx files with rich formatting
- **JSON Parser**: Handle nested JSON structures and arrays
- **XML Parser**: Parse structured XML data with schema validation
- **Web Scraper Parser**: Extract content from web pages and APIs
- **Email Parser**: Process .eml and .msg files with attachment handling
- **Markdown Parser**: Parse .md files with proper heading hierarchy

#### Specialized Parsers
- **Code Parser**: Extract functions, classes, and documentation from source code
- **Log Parser**: Structure log files with timestamp and severity extraction
- **Database Parser**: Connect directly to databases for real-time ingestion

### 🗄️ New Vector Database Support

#### Enterprise Vector Stores
- **Pinecone**: Managed vector database with high-performance search
- **Weaviate**: GraphQL-based vector database with semantic capabilities
- **Qdrant**: High-performance vector similarity search engine
- **Milvus**: Open-source vector database for large-scale deployments
- **LanceDB**: Serverless vector database with SQL support

#### Traditional Database Integration
- **PostgreSQL with pgvector**: Leverage existing PostgreSQL infrastructure
- **Elasticsearch**: Full-text search with vector similarity capabilities
- **Redis with RedisSearch**: In-memory vector search for low-latency applications

### 🧠 New Embedding Model Support

#### Commercial Embedding APIs
- **OpenAI Embeddings**: text-embedding-ada-002 and newer models
- **Cohere Embed**: Multilingual embedding models
- **Anthropic Claude**: When embedding APIs become available
- **Google PaLM Embeddings**: Google's large language model embeddings

#### Open Source Models
- **Sentence Transformers**: Wide variety of pre-trained models
- **Hugging Face Transformers**: Custom and fine-tuned embedding models
- **BGE Models**: Beijing Academy of AI's high-performance embedders
- **E5 Models**: Microsoft's multilingual embedding models

#### Specialized Embeddings
- **Code Embeddings**: Models trained specifically for source code
- **Domain-Specific Models**: Healthcare, legal, finance-trained embedders
- **Multimodal Embeddings**: Support for text + image embeddings

### 🤖 AI Integration & API Keys

#### ChatGPT/OpenAI Integration
- **API Key Management**: Secure storage and rotation of OpenAI API keys
- **GPT Integration**: Use GPT models for query expansion and result summarization
- **Function Calling**: Leverage GPT's function calling for structured queries
- **Streaming Responses**: Real-time response streaming for better UX

#### Multi-Provider Support
- **Anthropic Claude**: API key management and integration
- **Google Gemini**: Support for Google's AI models
- **Azure OpenAI**: Enterprise OpenAI service integration
- **Local LLM Integration**: Support for locally hosted language models

### 🎨 User Experience Enhancements

#### Advanced CLI Features
- **Interactive Mode**: Step-by-step guided setup and usage
- **Configuration Wizard**: Automated configuration generation
- **Batch Processing**: Handle multiple files and directories
- **Progress Persistence**: Resume interrupted processing
- **Result Export**: Save search results in various formats

#### Web Interface
- **REST API**: HTTP endpoints for all RAG operations
- **Web Dashboard**: Browser-based interface for management
- **Real-time Monitoring**: Live system status and performance metrics
- **User Authentication**: Multi-user support with role-based access

### 🔒 Security & Compliance

#### Data Protection
- **Encryption at Rest**: Secure storage of documents and embeddings
- **API Key Encryption**: Secure credential management
- **Access Controls**: Fine-grained permissions for different operations
- **Audit Logging**: Complete audit trail of all system operations

#### Privacy Features
- **Data Anonymization**: Remove PII from documents before processing
- **Selective Deletion**: Remove specific documents from vector stores
- **Compliance Reporting**: Generate reports for regulatory requirements

### 📊 Performance & Scalability

#### Optimization Features
- **Distributed Processing**: Scale across multiple machines
- **GPU Acceleration**: Leverage CUDA for faster embedding generation
- **Caching Layer**: Intelligent caching of embeddings and results
- **Load Balancing**: Distribute requests across multiple instances

#### Monitoring & Analytics
- **Performance Metrics**: Track processing speed and accuracy
- **Usage Analytics**: Understand system usage patterns
- **Cost Tracking**: Monitor API usage and associated costs
- **Health Checks**: Automated system health monitoring

### 🧪 Advanced Features

#### Intelligent Processing
- **Semantic Chunking**: Smart document splitting based on content structure
- **Query Expansion**: Automatic query enhancement for better results
- **Result Ranking**: ML-based relevance scoring and ranking
- **Duplicate Detection**: Identify and handle duplicate content

#### Integration Capabilities
- **Webhook Support**: Real-time notifications for processing events
- **ETL Pipeline Integration**: Connect with data pipeline tools
- **Message Queue Support**: Async processing with RabbitMQ/Kafka
- **Kubernetes Deployment**: Cloud-native deployment configurations

Each of these enhancements builds upon the solid foundation already established, utilizing the factory pattern and modular architecture to ensure seamless integration and minimal breaking changes.

## License

This project is designed as a simple, extensible foundation for RAG systems. Feel free to use, modify, and extend as needed.