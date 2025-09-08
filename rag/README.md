# RAG System - Complete Documentation

A powerful, extensible RAG (Retrieval-Augmented Generation) system featuring **strategy-first configuration**, **hash-based deduplication**, and **modular architecture**. Built for developers who want production-ready document processing with minimal setup.

## 🌟 Key Features

- **🎯 Strategy-First Design**: Configure entire pipelines through YAML strategies
- **🔍 Advanced Retrieval**: Multiple retrieval strategies (similarity, reranked, filtered, hybrid)
- **🚫 Deduplication System**: Hash-based document and chunk deduplication
- **📊 Document Management**: Full CRUD operations with version tracking
- **🔧 Modular Components**: Pluggable parsers, extractors, embedders, and stores
- **💻 CLI-First**: Comprehensive command-line interface for all operations
- **🧹 Automatic Cleanup**: Built-in collection management and cleanup

## 📚 Documentation Structure

- **[Quick Start](#-quick-start)** - Get running in 2 minutes
- **[CLI Guide](cli/README.md)** - Complete CLI documentation
- **[Demos Guide](demos/README.md)** - Interactive demonstrations
- **[Strategy System](docs/STRATEGY_SYSTEM.md)** - Configuration and strategies
- **[Advanced Usage](docs/ADVANCED_USAGE.md)** - Production deployment
- **[API Reference](docs/API_REFERENCE.md)** - Programmatic usage
- **[Component Guide](docs/COMPONENTS.md)** - Parsers, extractors, stores

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **UV** (recommended) or pip - [UV Installation](https://docs.astral.sh/uv/getting-started/installation/)
- **Ollama** (for local embeddings) - [Download](https://ollama.com/download)

### 1. Installation (1 minute)

```bash
# Clone the repository
git clone <repository-url>
cd rag/

# Option 1: Using UV (recommended)
uv sync

# Option 2: Using pip
pip install -r requirements.txt

# Setup Ollama
ollama serve                     # Start in separate terminal
ollama pull nomic-embed-text     # Download embedding model
```

### 2. Test Installation

```bash
# Run system test
python cli.py test

# Expected output:
# ✅ CLI is working!
# ✅ Configuration loaded successfully
# ✅ Ollama embedder is accessible
```

### 3. Run Your First Demo (30 seconds)

```bash
# Interactive demo menu
python demos/run_all_cli_demos.py

# Or run a specific demo
python demos/demo1_research_papers_cli.py
```

### 4. Basic CLI Usage

```bash
# Ingest documents
python cli.py ingest path/to/documents --strategy research_papers_demo

# Search
python cli.py search "your query" --strategy research_papers_demo

# View collection info
python cli.py info --strategy research_papers_demo

# Clean up
python cli.py manage delete --all --strategy research_papers_demo
```

## 📖 Core Concepts

### Strategy System
The RAG system uses a **strategy-first** approach where entire pipelines are configured through YAML files:

```yaml
research_papers_demo:
  description: "Optimized for academic papers"
  parsers:
    - type: PDFParser
    - type: TextParser
  extractors:
    - type: HeadingExtractor
    - type: KeywordExtractor
  embedder:
    type: OllamaEmbedder
    model: nomic-embed-text
  vector_store:
    type: ChromaStore
    collection_name: research_papers
```

[Learn more about strategies →](docs/STRATEGY_SYSTEM.md)

### Deduplication System
Built-in hash-based deduplication prevents duplicate storage:

- **Document-level hashing**: Entire documents tracked by content hash
- **Chunk-level hashing**: Individual chunks have unique IDs
- **Source tracking**: Files tracked to prevent re-ingestion

[Learn more about deduplication →](docs/DEDUPLICATION.md)

### Component Architecture

```
Input → Parser → Extractor → Embedder → Vector Store → Retriever
         ↓          ↓           ↓            ↓             ↓
     [Documents] [Metadata] [Vectors]  [Storage]    [Results]
```

[Learn more about components →](docs/COMPONENTS.md)

## 🎯 Common Use Cases

### 1. Research Paper Analysis
```bash
# Configure for academic papers
python cli.py ingest papers/ --strategy research_papers_demo
python cli.py search "transformer architecture" --strategy research_papers_demo
```

### 2. Customer Support System
```bash
# Process support tickets
python cli.py ingest tickets.csv --strategy customer_support_demo
python cli.py search "password reset" --strategy customer_support_demo
```

### 3. Code Documentation
```bash
# Index code documentation
python cli.py ingest docs/ --strategy code_documentation_demo
python cli.py search "API authentication" --strategy code_documentation_demo
```

### 4. Document Management
```bash
# Manage collections
python cli.py manage stats --strategy my_strategy
python cli.py manage delete --older-than 30 --strategy my_strategy
python cli.py manage delete --all --strategy my_strategy
```

## 🔧 CLI Commands Overview

| Command | Description | Example |
|---------|-------------|---------|
| `test` | Test system setup | `python cli.py test` |
| `ingest` | Add documents to collection | `python cli.py ingest path/ --strategy demo` |
| `search` | Search documents | `python cli.py search "query" --strategy demo` |
| `info` | Show collection information | `python cli.py info --strategy demo` |
| `manage` | Collection management | `python cli.py manage delete --all --strategy demo` |
| `strategies` | List/show strategies | `python cli.py strategies list` |

[Complete CLI Reference →](cli/README.md)

## 📊 Demos

The system includes 6 comprehensive demos:

1. **Research Papers** - Academic document processing
2. **Customer Support** - Ticket analysis and routing
3. **Code Documentation** - Technical documentation search
4. **News Analysis** - News article processing with entity extraction
5. **Business Reports** - Financial document analysis
6. **Document Management** - Advanced collection operations

[Run Demos →](demos/README.md)

## 🏗️ Project Structure

```
rag/
├── cli.py                  # Main CLI entry point
├── core/                   # Core abstractions
│   ├── base.py            # Base classes
│   ├── config.py          # Configuration management
│   ├── document_manager.py # Document lifecycle
│   └── strategies.py      # Strategy system
├── components/            # Pluggable components
│   ├── parsers/          # Document parsers
│   ├── extractors/       # Metadata extractors
│   ├── embedders/        # Embedding models
│   ├── stores/           # Vector databases
│   └── retrievers/       # Search strategies
├── demos/                # Demo applications
│   ├── demo_strategies.yaml # Demo configurations
│   └── static_samples/   # Sample data
├── utils/                # Utilities
│   └── hash_utils.py     # Deduplication system
└── docs/                 # Documentation
```

## 🔌 Programmatic Usage

```python
from core.strategies import StrategyManager
from core.config import load_config

# Load strategy
manager = StrategyManager(strategies_file="demo_strategies.yaml")
config = manager.get_strategy_config("research_papers_demo")

# Create pipeline
pipeline = create_pipeline_from_config(config)

# Process documents
results = pipeline.ingest("path/to/documents")

# Search
search_results = pipeline.search("your query", top_k=5)
```

[API Reference →](docs/API_REFERENCE.md)

## 🛠️ Configuration

### Environment Variables
```bash
export OLLAMA_HOST=http://localhost:11434
export CHROMA_PERSIST_DIR=./vectordb
export LOG_LEVEL=INFO
```

### Custom Strategies
Create your own strategies in YAML:

```yaml
my_custom_strategy:
  description: "My custom configuration"
  parsers:
    - type: PDFParser
      config:
        extract_images: true
  extractors:
    - type: CustomExtractor
      config:
        patterns: ["\\d{4}-\\d{2}-\\d{2}"]
  # ... more configuration
```

[Configuration Guide →](docs/CONFIGURATION.md)

## 🧪 Testing

The RAG system includes comprehensive test coverage with 40+ test files covering all components:

```bash
# Run all tests (basic command)
pytest

# Run with coverage reporting
pytest --cov

# Run fast tests only (skip slow/integration)
pytest -m "not slow and not integration"

# Run specific test categories
pytest -m integration              # Integration tests only
pytest -m "not ollama"            # Skip tests requiring Ollama
pytest tests/test_retrieval_system.py  # Specific test file

# Generate HTML coverage report
pytest --cov --cov-report=html
open htmlcov/index.html
```

**Test Organization:**
- `tests/components/` - Unit tests for parsers, extractors, embedders, stores
- `tests/e2e/` - End-to-end integration tests
- `tests/test_data/` - Test fixtures and sample documents

[Complete testing guide →](tests/README.md)

## 📋 Recent Improvements

### Schema & Configuration System
- **✅ Cleaned Schema**: Removed 500+ lines of legacy/duplicate code from `schema.yaml`
- **✅ Fixed Nested Structure**: Eliminated confusing `rag.rag` nesting in configurations
- **✅ Component Validation**: Schema now reflects actual component implementations
- **✅ Backward Compatibility**: Removed deprecated fields and legacy sections

### Enhanced Testing
- **✅ Comprehensive Coverage**: 40+ test files with >90% coverage for core modules
- **✅ Test Documentation**: Complete testing guide with commands and best practices
- **✅ Smart Test Markers**: Skip slow/integration tests during development
- **✅ Service Mocking**: Tests work without external dependencies

### Configuration Templates
- **✅ Clean Structure**: Fixed indentation and structure in `default.yaml`
- **✅ Validated Configs**: All templates validated against cleaned schema
- **✅ Component Accuracy**: Only references actual, implemented components

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install dev dependencies
uv sync --dev

# Run pre-commit hooks
pre-commit install

# Run linting
ruff check .
black .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full Docs](docs/)
- **Issues**: [GitHub Issues](https://github.com/...)
- **Discussions**: [GitHub Discussions](https://github.com/...)

## 🎉 Acknowledgments

Built with:
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Ollama](https://ollama.com/) - Local LLM embeddings
- [LangChain](https://langchain.com/) - Optional framework support
- [Rich](https://github.com/Textualize/rich) - Terminal formatting

---

**Ready to build?** Start with the [CLI Guide](cli/README.md) or jump into the [Demos](demos/README.md)!