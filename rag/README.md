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

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for embeddings)

### macOS Installation with UV (Recommended)

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

1. **Create configuration file:**
   Create a `rag_config.json` file with your settings (see Configuration section below).

2. **Test the system:**
   ```bash
   uv run python cli.py test --test-file filtered-english-incident-tickets.csv
   ```

3. **Ingest your CSV data:**
   ```bash
   uv run python cli.py ingest filtered-english-incident-tickets.csv
   ```

4. **Search the data:**
   ```bash
   uv run python cli.py search "data breach security issue"
   uv run python cli.py search "login problems"
   uv run python cli.py search "password reset"
   ```

5. **Get collection info:**
   ```bash
   uv run python cli.py info
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
uv run python example.py

# Or with activated environment
python example.py
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

### 2. Embedders
Generate vector embeddings from text content.

- `OllamaEmbedder`: Uses local Ollama models

### 3. Vector Stores
Store and search document embeddings.

- `ChromaStore`: ChromaDB integration with persistence

### 4. Pipeline
Chains components together for end-to-end processing.

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

- `config_examples/basic_config.json`: Basic local setup
- `config_examples/production_config.json`: Production deployment
- `config_examples/custom_csv_config.json`: Custom CSV format

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
python test_system.py
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
│   └── base.py              # Base classes
├── parsers/
│   ├── __init__.py
│   └── csv_parser.py        # CSV parsers
├── embedders/
│   ├── __init__.py
│   └── ollama_embedder.py   # Ollama integration
├── stores/
│   ├── __init__.py
│   └── chroma_store.py      # ChromaDB integration
├── config_examples/         # Sample configurations
├── cli.py                   # Command-line interface
├── example.py              # Usage example
├── test_system.py          # Test suite
├── requirements.txt        # Dependencies
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
uv run python test_system.py

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
- **PDF Parser**: Extract text, metadata, and structure from PDF documents
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