# RAG System Developer Architecture Guide

This document is the comprehensive developer reference for understanding and extending the RAG system architecture. It covers the system structure, component organization, and provides detailed guidance for adding new components.

## 🏗️ System Architecture Overview

The RAG system follows a **modular, strategy-first architecture** designed for:
- **Component Extensibility**: Easy addition of new parsers, embedders, stores, and extractors
- **Strategy-Based Configuration**: Pre-configured combinations for common use cases
- **Local-First Philosophy**: Prioritize local execution with optional cloud providers
- **Factory Pattern**: Centralized component creation and registration
- **Schema Validation**: Pydantic-based configuration validation

## 📁 Directory Structure

```
rag/
├── 🧩 components/              # Modular RAG components (new architecture)
│   ├── embedders/              # Embedding providers with schemas
│   │   ├── ollama_embedder/    # Local Ollama integration
│   │   ├── openai_embedder/    # OpenAI embeddings API
│   │   ├── huggingface_embedder/ # HuggingFace transformers
│   │   └── sentence_transformer_embedder/ # SentenceTransformers
│   ├── extractors/             # Content analysis extractors
│   │   ├── entity_extractor/   # Named entity recognition
│   │   ├── keyword_extractor/  # YAKE, RAKE, TF-IDF keywords
│   │   ├── summary_extractor/  # Text summarization
│   │   ├── statistics_extractor/ # Readability metrics
│   │   ├── table_extractor/    # Table detection/extraction
│   │   ├── link_extractor/     # URL/email/phone extraction
│   │   ├── heading_extractor/  # Document structure
│   │   ├── pattern_extractor/  # Regex-based patterns
│   │   └── date_time_extractor/ # Date/time extraction
│   ├── parsers/                # Document format parsers
│   │   ├── text_parser/        # Plain text documents
│   │   ├── csv_parser/         # CSV with specialized variants
│   │   ├── pdf_parser/         # PDF with metadata extraction
│   │   ├── docx_parser/        # Word documents
│   │   ├── excel_parser/       # Excel spreadsheets
│   │   ├── html_parser/        # HTML content cleaning
│   │   └── markdown_parser/    # Markdown structure preservation
│   ├── retrievers/             # Advanced retrieval strategies
│   │   ├── basic_similarity/   # Vector similarity search
│   │   ├── metadata_filtered/  # Metadata-aware filtering
│   │   ├── multi_query/        # Query expansion strategies
│   │   ├── reranked/          # Multi-factor re-ranking
│   │   └── hybrid_universal/   # Strategy combinations
│   ├── stores/                 # Vector database integrations
│   │   ├── chroma_store/       # ChromaDB (local development)
│   │   ├── faiss_store/        # FAISS (high performance)
│   │   ├── pinecone_store/     # Pinecone (managed cloud)
│   │   └── qdrant_store/       # Qdrant (self-hosted)
│   └── metadata/               # Metadata management
│       ├── metadata_config.py  # Schema definitions
│       └── metadata_enricher.py # Automatic enrichment
├── 🏛️ core/                   # Core system architecture
│   ├── base.py                 # Abstract base classes
│   ├── factories.py            # Component factories & registration
│   ├── document_manager.py     # Document lifecycle management
│   ├── enhanced_pipeline.py    # Pipeline with progress tracking
│   └── extractor_integration.py # Extractor orchestration
├── 🎛️ strategies/              # Strategy system (new!)
│   ├── manager.py              # Strategy loading and validation
│   ├── loader.py              # YAML strategy file loading
│   └── config.py              # Strategy configuration models
├── 🎭 demos/                   # 5 comprehensive demos + utilities
│   ├── demo1_research_papers.py      # Academic content analysis
│   ├── demo2_customer_support.py     # Support ticket system
│   ├── demo3_code_documentation.py   # Technical documentation
│   ├── demo4_news_analysis.py        # Media content processing
│   ├── demo5_business_reports.py     # Financial document analysis
│   ├── master_demo.py                # Runs all 5 demos
│   ├── enhanced_demo.py              # Detailed embedding showcase
│   ├── demo_strategies.yaml          # Demo strategy configurations
│   ├── strategy_demo_utils.py        # Strategy-based demo utilities
│   ├── utils.py                      # Shared demo utilities
│   ├── static_samples/               # Sample documents for demos
│   │   ├── research_papers/          # Academic texts
│   │   ├── customer_support/         # Support tickets & KB
│   │   ├── code_documentation/       # API docs & guides
│   │   ├── news_articles/            # HTML news content
│   │   └── business_reports/         # Financial documents (PDF/Excel)
│   └── vectordb/                     # Demo vector databases
│       ├── research_papers/          # Research demo ChromaDB
│       ├── customer_support/         # Support demo ChromaDB
│       └── [other demo collections]
├── 🧪 tests/                   # Comprehensive test suite
│   ├── components/             # Component-specific tests
│   │   ├── embedders/         # Embedder testing with mocks
│   │   ├── extractors/        # Extractor functionality tests
│   │   ├── parsers/           # Parser format validation
│   │   └── stores/            # Vector store integration tests
│   ├── conftest.py            # Shared test fixtures
│   └── [legacy test files]    # Compatibility tests
├── 📊 schemas/                 # Generated component schemas
│   ├── embedders.json         # Embedder type schemas
│   ├── extractors.json        # Extractor configuration schemas
│   ├── parsers.json           # Parser option schemas
│   ├── stores.json           # Vector store schemas
│   └── strategies.json        # Strategy validation schemas
├── 📋 examples/                # Usage examples and configs
│   ├── configs/               # Example configurations
│   ├── example.py             # Basic usage example
│   ├── example_api_usage.py   # Internal API usage
│   └── test_system.py         # System integration test
├── 🛠️ utils/                   # System utilities
│   ├── path_resolver.py       # Flexible path resolution
│   └── progress.py           # Progress tracking with style
├── 📄 config/                  # Default configurations
│   └── default.yaml           # System defaults
├── 📜 scripts/                 # Utility scripts
│   ├── create_test_pdf.py     # Test document generation
│   └── quick_extractor_demo.sh # Extractor testing
├── 🚀 api.py                   # Internal search API
├── 🖥️ cli.py                   # Full-featured CLI interface
├── 🔧 cli_compile.py           # Schema compilation utility
├── ⚙️ default_strategies.yaml  # Built-in strategy definitions
├── 📝 schema.yaml             # System schema definition
└── 📖 [documentation files]   # README.md, STRUCTURE.md, etc.
```

## 🧩 Components Overview

### Embedders (`components/embedders/`)
- **OllamaEmbedder**: Local embedding using Ollama models
- **OpenAIEmbedder**: OpenAI's embedding API
- **HuggingFaceEmbedder**: HuggingFace transformer models
- **SentenceTransformerEmbedder**: Sentence transformer models

### Extractors (`components/extractors/`)
- **keyword_extractors**: YAKE, RAKE, TF-IDF keyword extraction
- **entity_extractor**: Named entity recognition
- **table_extractor**: Table detection and extraction
- **link_extractor**: URL, email, and link extraction
- **heading_extractor**: Document structure and headings
- **statistics_extractor**: Text statistics and readability metrics
- **summary_extractor**: Automatic text summarization
- **pattern_extractor**: Regex-based pattern extraction

### Parsers (`components/parsers/`)
- **text_parser**: Plain text documents
- **markdown_parser**: Markdown files with structure extraction
- **html_parser**: HTML documents with content cleaning
- **pdf_parser**: PDF documents (requires pypdf)
- **csv_parser**: CSV files with various formats
- **docx_parser**: Word documents (requires python-docx)
- **excel_parser**: Excel files (requires pandas/openpyxl)

### Vector Stores (`components/stores/`)
- **chroma_store**: ChromaDB for local vector storage
- **faiss_store**: FAISS for high-performance similarity search
- **pinecone_store**: Pinecone cloud vector database
- **qdrant_store**: Qdrant vector database

### Retrievers (`components/retrievers/`)
- **strategies/**: Various retrieval strategies
- **factory.py**: Retrieval strategy factory
- **registry.py**: Strategy registration and management

## 🚀 Quick Start

### 1. Run the Enhanced Demo
```bash
./run_demo.sh
```

This demo shows:
- Real document parsing from multiple formats
- Actual embedding generation with detailed vector information
- Vector storage in ChromaDB
- Semantic search with similarity scores
- Complete transparency in the RAG process

### 2. Use the CLI
```bash
# Search documents
uv run python cli.py search "machine learning basics"

# List available components
uv run python cli.py extractors list
uv run python cli.py strategies list

# Ingest your own documents
uv run python cli.py ingest /path/to/your/documents
```

### 3. Run Tests
```bash
# Test all components
uv run python -m pytest tests/

# Test specific components
uv run python tests/test_new_parsers.py
uv run python tests/test_new_extractors.py
```

## 📚 Sample Documents

The `samples/documents/` directory contains real, comprehensive documents:

- **ai_overview.txt**: Complete overview of AI technologies
- **machine_learning_guide.md**: ML implementation guide
- **data_science_workflow.txt**: End-to-end data science process
- **python_programming_basics.md**: Python fundamentals

These documents provide rich, realistic content for testing and demonstration.

## 🎯 Key Features

### ✨ Real Embedding Demonstration
- Shows actual embedding vectors being generated
- Displays embedding dimensions and sample values
- Demonstrates semantic similarity through vector space

### 🔍 Transparent Retrieval
- Shows query embeddings vs document embeddings
- Displays similarity scores for search results
- Provides detailed result analysis

### 🧹 Clean Architecture
- Components organized by functionality
- Clear separation of concerns
- Easy to extend and maintain

### 🔧 Flexible Configuration
- Strategy-based approach for different use cases
- Configurable components and parameters
- Support for multiple data formats and stores

## 🔧 Adding New Components

### Adding a New Parser

1. **Create parser directory and files:**
   ```bash
   mkdir -p components/parsers/your_parser/
   touch components/parsers/your_parser/__init__.py
   touch components/parsers/your_parser/your_parser.py
   touch components/parsers/your_parser/schema.py
   ```

2. **Implement the parser class:**
   ```python
   # components/parsers/your_parser/your_parser.py
   from typing import List, Any, Dict
   from core.base import BaseParser, Document

   class YourParser(BaseParser):
       """Parser for your specific format."""

       def __init__(self, config: Dict[str, Any]):
           super().__init__(config)
           # Initialize format-specific settings

       def parse(self, content: bytes, metadata: Dict[str, Any] = None) -> List[Document]:
           """Parse content into Document objects."""
           # Implement your parsing logic
           return [Document(content=text, metadata=metadata or {})]
   ```

3. **Define configuration schema:**
   ```python
   # components/parsers/your_parser/schema.py
   from pydantic import BaseModel, Field
   from typing import Optional

   class YourParserConfig(BaseModel):
       """Configuration for YourParser."""
       option1: bool = Field(True, description="Enable option 1")
       option2: Optional[str] = Field(None, description="Optional setting")
   ```

4. **Register with factory:**
   ```python
   # Add to core/factories.py
   from components.parsers.your_parser.your_parser import YourParser

   class ParserFactory(ComponentFactory):
       _registry = {
           # ... existing parsers
           "YourParser": YourParser,
       }
   ```

5. **Write comprehensive tests:**
   ```python
   # tests/components/parsers/test_your_parser.py
   import pytest
   from components.parsers.your_parser.your_parser import YourParser

   class TestYourParser:
       def test_basic_parsing(self):
           parser = YourParser(name="YourParser", config={"option1": True})
           # Test parsing logic
   ```

### Adding a New Extractor

1. **Create extractor structure:**
   ```bash
   mkdir -p components/extractors/your_extractor/
   touch components/extractors/your_extractor/__init__.py
   touch components/extractors/your_extractor/your_extractor.py
   touch components/extractors/your_extractor/schema.py
   ```

2. **Implement extractor class:**
   ```python
   # components/extractors/your_extractor/your_extractor.py
   from typing import Dict, Any, List
   from core.base import BaseExtractor, Document

   class YourExtractor(BaseExtractor):
       """Extract specific information from documents."""

       def extract(self, document: Document) -> Dict[str, Any]:
           """Extract data and return as metadata."""
           extracted_data = {
               "extracted_field": "value",
               "confidence_score": 0.95
           }
           return extracted_data
   ```

3. **Register and test following similar patterns to parsers**

### Adding a New Vector Store

1. **Create store structure:**
   ```bash
   mkdir -p components/stores/your_store/
   touch components/stores/your_store/__init__.py
   touch components/stores/your_store/your_store.py
   ```

2. **Implement store class:**
   ```python
   # components/stores/your_store/your_store.py
   from typing import List, Dict, Any, Optional
   from core.base import BaseVectorStore, Document

   class YourStore(BaseVectorStore):
       """Vector store implementation for your database."""

       def add_documents(self, documents: List[Document]) -> List[str]:
           """Add documents and return IDs."""
           # Implement storage logic

       def similarity_search(self, query: str, k: int = 4,
                           filter: Optional[Dict[str, Any]] = None) -> List[Document]:
           """Search for similar documents."""
           # Implement search logic
   ```

### Adding a New Retrieval Strategy

1. **Create strategy:**
   ```python
   # components/retrievers/your_strategy/your_strategy.py
   from typing import List, Dict, Any
   from core.base import BaseRetrievalStrategy, Document

   class YourStrategy(BaseRetrievalStrategy):
       """Your custom retrieval strategy."""

       def retrieve(self, query: str, vector_store, **kwargs) -> List[Document]:
           """Implement your retrieval logic."""
           # Custom retrieval implementation
   ```

## 🏗️ Architecture Patterns

### Factory Pattern
The system uses factories to create components dynamically:
- **Centralized Registration**: All components register with their respective factories
- **Configuration-Driven**: Components created from YAML/JSON configurations
- **Type Safety**: Pydantic validation ensures correct configuration
- **Extensibility**: Easy to add new component types without changing core code

### Strategy System
**Strategy-first approach** for common use cases:
```yaml
# Strategy definition in strategies/
research_strategy:
  description: "Optimized for academic research"
  components:
    parser:
      type: "PlainTextParser"
      config:
        chunk_size: 1000
    extractors:
      - type: "StatisticsExtractor"
      - type: "EntityExtractor"
    embedder:
      type: "OllamaEmbedder"
      config:
        model: "nomic-embed-text"
```

### Configuration Flow
1. **Strategy Loading**: `strategies/manager.py` loads YAML strategies
2. **Component Creation**: `core/factories.py` creates components from config
3. **Pipeline Assembly**: `core/enhanced_pipeline.py` orchestrates processing
4. **Schema Validation**: All configs validated with Pydantic models

## 📊 Testing Guidelines

### Component Testing Structure
```python
# tests/components/[type]/test_[component].py
class TestYourComponent:
    @pytest.fixture
    def config(self):
        return {"param": "value"}

    @pytest.fixture
    def component(self, config):
        return YourComponent(config)

    def test_initialization(self, component):
        assert component is not None

    def test_functionality(self, component):
        # Test core functionality
        pass

    @patch('external.service')
    def test_external_dependencies(self, mock_service, component):
        # Mock external services
        pass
```

### Integration Testing
- **End-to-end pipeline tests**: Full document processing workflows
- **Strategy validation**: Ensure strategies work with real components
- **Performance benchmarks**: Track processing speed and memory usage
- **Mock external services**: Test without external dependencies

### Running Tests
```bash
# All tests
uv run python -m pytest tests/

# Component-specific tests
uv run python -m pytest tests/components/parsers/
uv run python -m pytest tests/components/extractors/

# Integration tests
uv run python -m pytest tests/test_retrieval_system.py

# With coverage
uv run python -m pytest --cov=components --cov-report=html
```

## 🎯 Best Practices

### Component Design
- **Single Responsibility**: Each component has one clear purpose
- **Interface Compliance**: Follow base class contracts strictly
- **Error Handling**: Graceful degradation with informative errors
- **Resource Management**: Clean up resources (files, connections)
- **Configuration Validation**: Use Pydantic for type safety

### Performance Optimization
- **Batch Processing**: Process multiple documents efficiently
- **Memory Management**: Stream large files, don't load entirely
- **Caching**: Cache expensive operations when appropriate
- **Async Support**: Use async for I/O-bound operations
- **Resource Pooling**: Reuse expensive resources

### Development Workflow
1. **Design**: Plan component interface and responsibilities
2. **Implement**: Follow existing patterns and conventions
3. **Test**: Write comprehensive tests first (TDD recommended)
4. **Integrate**: Add to factories and strategy definitions
5. **Document**: Update schemas and documentation
6. **Validate**: Run full test suite and demo system

This architecture provides a clean, maintainable, and easily extensible RAG system that demonstrates real-world functionality with complete transparency.
