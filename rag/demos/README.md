# RAG System Demos

This directory contains comprehensive demonstrations of the RAG (Retrieval-Augmented Generation) system, showcasing different strategies and optimizations for specific domains and use cases.

## 🎯 Demo Overview

Each demo demonstrates:
- Different parsing strategies for various document formats
- Specialized extractors optimized for domain-specific content
- Unique search and retrieval approaches
- Real-world applications and use cases

## 🚀 Quick Start

### Run All Demos (Interactive Menu)
```bash
# Interactive menu to run all demos or select specific ones
python demos/run_all_cli_demos.py

# Or using uv
uv run python demos/run_all_cli_demos.py
```

### Run Individual CLI Demos

```bash
# Demo 1: Research Papers Analysis (CLI)
python demos/demo1_research_papers_cli.py

# Demo 2: Customer Support System (CLI)
python demos/demo2_customer_support_cli.py

# Demo 3: Code Documentation (CLI)
python demos/demo3_code_documentation_cli.py

# Demo 4: News Article Analysis
python demos/demo4_news_analysis.py

# Demo 5: Business Reports Analysis
python demos/demo5_business_reports.py

# Demo 6: Document Management & Vector DB Operations
python demos/demo6_document_management.py
```

## 📚 Available Demos

### Demo 1: Research Papers Analysis
- **Strategy**: `research_papers_demo`
- **Features**: Directory parsing, PDF/TXT/MD files, academic search
- **Highlights**: Shows verbose embeddings, metadata extraction, entity recognition
- **Key Extractors**: ContentStatisticsExtractor, EntityExtractor, SummaryExtractor
- **Use Cases**: Literature reviews, research synthesis, evidence discovery

### Demo 2: Customer Support System
- **Strategy**: `customer_support_demo`
- **Features**: CSV & TXT parsing, ticket analysis, priority detection
- **Highlights**: Metadata filtering, sentiment analysis, knowledge base integration
- **Key Extractors**: EntityExtractor, PatternExtractor, SummaryExtractor
- **Use Cases**: Support case resolution, knowledge base search, agent assistance

### Demo 3: Code Documentation Analysis
- **Strategy**: `code_documentation_demo`
- **Features**: Markdown/Python parsing, code extraction, technical search
- **Highlights**: HybridUniversalStrategy, code pattern extraction
- **Key Extractors**: HeadingExtractor, LinkExtractor, PatternExtractor
- **Use Cases**: API documentation, developer guides, code examples

### Demo 4: News Article Analysis
- **Strategy**: `news_analysis_demo`
- **Features**: HTML/TXT parsing, entity extraction, temporal analysis
- **Highlights**: RerankedStrategy with recency boost, entity recognition
- **Key Extractors**: EntityExtractor, SummaryExtractor, LinkExtractor
- **Use Cases**: News analysis, trend identification, content insights

### Demo 5: Business Reports Analysis
- **Strategy**: `business_reports_demo`
- **Features**: PDF parsing, financial metrics extraction
- **Highlights**: Pattern extraction for financial data, MultiQueryStrategy
- **Key Extractors**: TableExtractor, ContentStatisticsExtractor, EntityExtractor
- **Use Cases**: Financial analysis, KPI tracking, business intelligence

### Demo 6: Document Management & Vector DB
- **Features**: Add/search/delete/replace operations, metadata queries
- **Highlights**: Document lifecycle, deduplication, hashing, collection management
- **Use Cases**: Collection management, document operations, metadata handling

## 🎨 CLI Commands Reference

### Basic Commands
```bash
# Initialize a collection
python cli.py --strategy-file demos/demo_strategies.yaml init --strategy <strategy_name>

# Ingest documents
python cli.py --strategy-file demos/demo_strategies.yaml ingest <path> --strategy <strategy_name>

# Search with verbose output
python cli.py --strategy-file demos/demo_strategies.yaml --verbose search "query" --strategy <strategy_name> --top-k 5

# Get collection info
python cli.py --strategy-file demos/demo_strategies.yaml info --strategy <strategy_name>

# Manage documents
python cli.py --strategy-file demos/demo_strategies.yaml manage --rag-strategy <strategy_name> stats
python cli.py --strategy-file demos/demo_strategies.yaml manage --rag-strategy <strategy_name> delete --doc-ids <id1> <id2>

# List available strategies
python cli.py strategies list

# Show strategy details
python cli.py strategies show <strategy_name>

# List available extractors
python cli.py extractors list

# Test system
python cli.py test
```

### Command Syntax Examples

```bash
# NOTE: Command syntax follows this pattern:
# python cli.py [global-options] <command> [positional-args] [command-options]

# Correct: Search query comes directly after 'search' command
python cli.py --strategy-file demos/demo_strategies.yaml search "transformer architecture" --strategy research_papers_demo

# Correct: Verbose is a global option (before command)
python cli.py --strategy-file demos/demo_strategies.yaml --verbose search "AI breakthrough" --strategy news_analysis_demo

# Correct: Quiet mode with custom content length
python cli.py --strategy-file demos/demo_strategies.yaml --quiet --content-length 200 search "revenue" --strategy business_reports_demo
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- Ollama installed and running
- nomic-embed-text model loaded in Ollama

### Install Ollama Model
```bash
ollama pull nomic-embed-text
```

### Verify Setup
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should show nomic-embed-text in the model list
```

## 📁 Directory Structure

```
demos/
├── README.md                          # This file
├── DEMO_README.md                     # Legacy documentation
├── run_all_cli_demos.py              # Interactive demo runner
├── demo1_research_papers_cli.py      # Research paper analysis
├── demo2_customer_support_cli.py     # Customer support system
├── demo3_code_documentation_cli.py   # Code documentation
├── demo4_news_analysis.py            # News article analysis
├── demo5_business_reports.py         # Business reports analysis
├── demo6_document_management.py      # Document management operations
├── demo_strategies.yaml              # All demo strategy configurations
├── static_samples/                   # Sample data for demos
│   ├── research_papers/              # Academic research content
│   ├── customer_support/             # Support tickets and knowledge base
│   ├── code_documentation/           # Technical documentation
│   ├── news_articles/                # News and media content
│   └── business_reports/             # Financial and business data
└── archive/                          # Archived demo files
```

## 🔧 100% Configuration-Driven Architecture

**IMPORTANT**: All demos use the RAG framework with **ZERO hardcoded configuration**:
- ✅ **100% Strategy-Based**: All configuration comes from `demo_strategies.yaml`
- ✅ **No Hardcoding**: Search parameters, component configs, extractors - all in YAML
- ✅ **Real Framework Usage**: Actual RAG components, not simulations
- ✅ **Production-Ready Code**: Same patterns used in enterprise deployments
- ✅ **Easy Customization**: Change behavior by editing YAML, not Python code

### How Configuration Works

All demos load their configuration from `demo_strategies.yaml`:

```python
# Example from demo code (NOT hardcoded values!)
from strategies import StrategyManager

# Load strategy configuration
strategy_manager = StrategyManager()
config = strategy_manager.convert_strategy_to_config("research_papers_demo")

# The RAG system uses this configuration for ALL components
rag_system = RAGSystem(config)
```

To modify demo behavior:
1. Edit `demo_strategies.yaml` - change extractors, parsers, retrieval settings
2. No Python code changes needed - all behavior is configuration-driven
3. Add new extractors by listing them in the YAML configuration
4. Adjust search parameters (top_k, distance_metric) in the strategy config

## 🔍 Querying Demo Databases

After running demos, you can query the created databases using the CLI:

```bash
# Research papers
python cli.py --strategy-file demos/demo_strategies.yaml search "transformer architecture" --strategy research_papers_demo

# Customer support
python cli.py --strategy-file demos/demo_strategies.yaml search "login issues" --strategy customer_support_demo

# Code documentation  
python cli.py --strategy-file demos/demo_strategies.yaml search "parser implementation" --strategy code_documentation_demo

# News analysis
python cli.py --strategy-file demos/demo_strategies.yaml search "AI quantum computing" --strategy news_analysis_demo

# Business reports
python cli.py --strategy-file demos/demo_strategies.yaml search "quarterly revenue" --strategy business_reports_demo
```

## 📊 What Each Demo Demonstrates

### Architecture Principles
- **Modularity**: Same components, different configurations
- **Specialization**: Domain-specific extractors and workflows
- **Scalability**: Isolated knowledge bases per domain
- **Consistency**: Unified API across all domains
- **Extensibility**: Easy to add new domains and extractors

### RAG Strategies
- **Format-specific parsing**: Optimized for different document types
- **Domain-aware extraction**: Specialized content analysis
- **Contextual embedding**: Semantic representation preservation
- **Targeted retrieval**: Optimized search for specific use cases
- **Intelligent analytics**: Domain-specific insights and metrics

### Performance Characteristics
- **Efficient processing**: Chunking and batch optimization
- **Memory management**: Streaming for large documents
- **Fast retrieval**: Domain-specific vector collections
- **Comprehensive analysis**: Parallel extractor processing
- **Scalable architecture**: Handles documents of any size

## ⚠️ Troubleshooting

### Common Issues

**Demo fails to start:**
- Ensure Ollama is running: `ollama serve`
- Check model is loaded: `ollama list`
- Verify Python dependencies: `uv sync` or `pip install -r requirements.txt`

**Embedding errors:**
- Restart Ollama service
- Check available memory (embeddings require RAM)
- Verify network connectivity to Ollama API

**Sample data not found:**
- Ensure all files in `static_samples/` directories exist
- Check file permissions are readable
- Verify working directory is correct (should be in `rag/` directory)

**Slow performance:**
- Reduce batch sizes in demo configurations
- Check available system memory
- Ensure Ollama has sufficient resources

### Performance Tips

- **Run demos sequentially** to avoid resource conflicts
- **Close other applications** to free up memory
- **Use SSD storage** for better vector database performance
- **Monitor system resources** during execution

## 🎓 Learning Outcomes

After running these demos, you'll understand:

1. **RAG Architecture Design**: How to structure modular, extensible RAG systems
2. **Domain Specialization**: Why and how to optimize for specific content types
3. **Extraction Strategies**: Different approaches for analyzing and enriching content
4. **Performance Optimization**: Techniques for efficient processing and retrieval
5. **Real-world Applications**: Practical use cases across multiple industries
6. **Configuration-First Development**: How to build systems where behavior is defined in YAML, not code
7. **CLI Design**: How to build intuitive command-line interfaces for RAG systems

## 💡 Extending the Demos

### Adding New Extractors

To add new extractors to any demo, update `demo_strategies.yaml`:

```yaml
strategy_name:
  components:
    extractors:
      - type: "YourExtractor"
        config:
          setting1: value1
          setting2: value2
```

The framework's factory pattern automatically loads any configured extractor. No code changes needed!

### Creating New Strategies

Add a new strategy to `demo_strategies.yaml`:

```yaml
your_new_strategy:
  description: "Your strategy description"
  use_cases: ["Use case 1", "Use case 2"]
  components:
    parser:
      type: "YourParser"
      config: {}
    extractors: []
    embedder:
      type: "OllamaEmbedder"
      config: {}
    vector_store:
      type: "ChromaStore"
      config: {}
    retrieval_strategy:
      type: "BasicSimilarityStrategy"
      config: {}
```

## 🤝 Contributing

Want to add more demos or improve existing ones?

1. **Follow the demo structure**: Parser → Extractors → Embedder → Vector Store
2. **Use strategy configuration**: All settings in `demo_strategies.yaml`
3. **Create comprehensive samples**: Realistic data for your domain
4. **Add detailed analysis**: Show what makes your approach unique
5. **Include educational content**: Explain why this strategy works
6. **Test thoroughly**: Ensure demos run reliably
7. **No hardcoding**: Keep all configuration in YAML files

## 📖 Further Documentation

For comprehensive CLI documentation, see [CLI Documentation](../docs/cli-guide.md).

## 📝 License

These demos are part of the RAG system project and follow the same licensing terms.