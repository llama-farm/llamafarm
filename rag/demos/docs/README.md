# RAG System Demos

This directory contains 5 comprehensive demonstrations of the RAG (Retrieval-Augmented Generation) system, each showcasing different strategies and optimizations for specific domains and use cases.

## 🎯 Demo Overview

Each demo is designed to **WOW** and **EDUCATE** by demonstrating:
- Different parsing strategies for various document formats
- Specialized extractors optimized for domain-specific content
- Unique search and retrieval approaches
- Real-world applications and use cases

## 🔧 100% Configuration-Driven Architecture

**IMPORTANT**: All demos use the RAG framework with **ZERO hardcoded configuration**:
- ✅ **100% Strategy-Based**: All configuration comes from `demo_strategies.yaml`
- ✅ **No Hardcoding**: Search parameters, component configs, extractors - all in YAML
- ✅ **Real Framework Usage**: Actual RAG components, not simulations
- ✅ **Production-Ready Code**: Same patterns used in enterprise deployments
- ✅ **Easy Customization**: Change behavior by editing YAML, not Python code

## 📚 Available Demos

### 1. Research Paper Analysis (`demo1_research_papers.py`)
**Domain**: Academic/Scientific Research  
**Strategy**: Statistical analysis + Citation extraction  
**Formats**: Plain text research papers  
**Extractors**: ContentStatisticsExtractor, EntityExtractor, SummaryExtractor  
**Use Cases**: Literature reviews, research synthesis, evidence discovery

**Configuration from `demo_strategies.yaml`:**
- PlainTextParser: 2000 char chunks with 300 overlap, preserves structure
- ContentStatisticsExtractor: Readability metrics, vocabulary analysis, sentiment indicators
- EntityExtractor: PERSON, ORG, GPE, DATE, PERCENT, PRODUCT entities
- SummaryExtractor: 3-sentence summaries with key phrases and statistics
- OllamaEmbedder: nomic-embed-text model, batch size 3
- ChromaStore: Persisted to `./demos/vectordb/research_papers`
- BasicSimilarityStrategy: Cosine similarity, top-k=3 results

**Why use this approach:**
- Extracts quantitative data automatically from research content
- Identifies key researchers, institutions, and methodologies
- Optimized for academic terminology and scientific concepts
- Enables semantic search across research corpus

### 2. Customer Support System (`demo2_customer_support.py`)
**Domain**: Business/Customer Service  
**Strategy**: Case matching + Resolution pattern recognition  
**Formats**: CSV (tickets) + Text (knowledge base)  
**Extractors**: EntityExtractor, PatternExtractor, SummaryExtractor  
**Use Cases**: Support case resolution, knowledge base search, agent assistance

**Configuration from `demo_strategies.yaml`:**
- CustomerSupportCSVParser: Extracts ticket metadata (id, customer, priority, category)
- Priority mapping: Critical=1, High=2, Medium=3, Low=4
- EntityExtractor: Customer names, products, dates mentioned
- PatternExtractor: Email, phone, URL, IP address patterns with context
- SummaryExtractor: 3-sentence issue summaries with key phrases
- OllamaEmbedder: Batch size 4 for faster processing
- ChromaStore: Persisted to `./demos/vectordb/customer_support`

**Why use this approach:**
- Automatic similar case detection and matching
- Resolution pattern recognition for consistent solutions
- Entity-aware issue categorization and routing
- Support analytics and trend identification

### 3. Code Documentation (`demo3_code_documentation.py`)
**Domain**: Technical/Software Development  
**Strategy**: Structure preservation + Cross-reference linking  
**Formats**: Markdown technical documentation  
**Extractors**: HeadingExtractor, LinkExtractor, PatternExtractor  
**Use Cases**: API documentation, developer guides, code examples

**Configuration from `demo_strategies.yaml`:**
- MarkdownParser: Preserves structure, extracts headers, handles code blocks
- HeadingExtractor: All heading levels (1-6) with content preview and hierarchy
- LinkExtractor: Internal/external links with anchor text (no validation)
- PatternExtractor: URL, email, and code reference patterns with context
- OllamaEmbedder: Standard batch processing for documentation
- ChromaStore: Persisted to `./demos/vectordb/code_documentation`

**Why use this approach:**
- Preserves code formatting and technical structure
- Maintains cross-reference relationships and navigation
- Optimized for technical terminology and concepts
- Context-aware code pattern matching

### 4. News Article Analysis (`demo4_news_analysis.py`)
**Domain**: Media/Journalism  
**Strategy**: Entity recognition + Trend tracking  
**Formats**: HTML news articles  
**Extractors**: Entity, Summary, Link  
**Use Cases**: News analysis, trend identification, content insights

**Configuration from `demo_strategies.yaml`:**
- HTMLParser: Configured to extract clean text, preserve links
- EntityExtractor: Recognizes PERSON, ORG, GPE, DATE, EVENT, MONEY entities
- SummaryExtractor: 2-sentence summaries with key phrases
- LinkExtractor: Captures both internal and external links with anchor text

**Why use this approach:**
- Identifies key entities and their relationships in news
- Maintains source verification and credibility links
- Enables topic clustering and trend analysis
- Ready for sentiment analysis when SentimentExtractor is added

**Note**: While the README mentioned sentiment analysis, the current `demo_strategies.yaml` uses EntityExtractor, SummaryExtractor, and LinkExtractor. A SentimentExtractor could be easily added to the configuration to enable sentiment tracking.

### 5. Business Reports Analysis (`demo5_business_reports.py`)
**Domain**: Financial/Corporate Intelligence  
**Strategy**: Multi-format processing + Metrics extraction  
**Formats**: Excel, PDF, CSV (mixed business data)  
**Extractors**: TableExtractor, ContentStatisticsExtractor, EntityExtractor, SummaryExtractor  
**Use Cases**: Financial analysis, KPI tracking, business intelligence

**Configuration from `demo_strategies.yaml`:**
- PDFParser: Extracts metadata, page structure, combines pages
- TableExtractor: Extracts tables with headers (min 2x2 size)
- ContentStatisticsExtractor: Readability, structure, and vocabulary metrics
- EntityExtractor: ORG, PERSON, MONEY, DATE, PERCENT entities
- SummaryExtractor: 3-sentence summaries with key phrases and statistics
- ChromaStore: Persisted to `./demos/vectordb/business_reports`

**Why use this approach:**
- Automatic financial metrics and KPI extraction
- Handles multiple business document formats
- Business trend identification through entity analysis
- Cross-report insights and correlation analysis

## 🚀 Quick Start

### Run Individual Demos

```bash
# Research paper analysis
uv run python demos/demo1_research_papers.py

# Customer support system
uv run python demos/demo2_customer_support.py

# Code documentation
uv run python demos/demo3_code_documentation.py

# News article analysis  
uv run python demos/demo4_news_analysis.py

# Business reports analysis
uv run python demos/demo5_business_reports.py
```

### Run All Demos (Master Showcase)

```bash
# Run the complete showcase (15-25 minutes)
uv run python demos/master_demo.py
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
├── master_demo.py                     # Runs all 5 demos
├── demo1_research_papers.py           # Research paper analysis
├── demo2_customer_support.py          # Customer support system
├── demo3_code_documentation.py        # Code documentation
├── demo4_news_analysis.py             # News article analysis
├── demo5_business_reports.py          # Business reports analysis
├── static_samples/                    # Sample data for demos
│   ├── research_papers/               # Academic research content
│   ├── customer_support/              # Support tickets and knowledge base
│   ├── code_documentation/            # Technical documentation
│   ├── news_articles/                 # News and media content
│   └── business_reports/              # Financial and business data
└── vectordb/                          # Generated vector databases
    ├── research_papers/               # Research paper embeddings
    ├── customer_support/              # Support case embeddings
    ├── code_documentation/            # Documentation embeddings
    ├── news_analysis/                 # News article embeddings
    └── business_reports/              # Business data embeddings
```

## 🎨 Demo Features

### Visual Experience
- Rich terminal output with colors and formatting
- Progress bars and real-time status updates
- Beautiful tables and panels for data presentation
- Emoji indicators for different content types

### Educational Content
- Detailed explanations of each strategy and approach
- Real-world use case demonstrations
- Performance metrics and analytics
- Comparative analysis across domains

### Technical Depth
- Shows actual embedding vectors and dimensions
- Demonstrates real retrieval with similarity scores
- Explains extractor results and data analysis
- Provides insights into RAG architecture decisions

## 🔍 Querying Demo Databases

After running demos, you can query the created databases:

```bash
# Research papers
uv run python cli.py search "transformer architecture performance" --collection research_papers

# Customer support
uv run python cli.py search "login issues password reset" --collection customer_support

# Code documentation  
uv run python cli.py search "parser implementation examples" --collection code_documentation

# News analysis
uv run python cli.py search "AI quantum computing trends" --collection news_articles

# Business reports
uv run python cli.py search "quarterly revenue growth trends" --collection business_reports
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
- Verify Python dependencies: `uv sync`

**Embedding errors:**
- Restart Ollama service
- Check available memory (embeddings require RAM)
- Verify network connectivity to Ollama API

**Sample data not found:**
- Ensure all files in `static_samples/` directories exist
- Check file permissions are readable
- Verify working directory is correct

**Slow performance:**
- Reduce batch sizes in demo configurations
- Check available system memory
- Ensure Ollama has sufficient resources

### Performance Tips

- **Run demos sequentially** to avoid resource conflicts
- **Close other applications** to free up memory
- **Use SSD storage** for better vector database performance
- **Monitor system resources** during execution

## 🔧 How Configuration Works

All demos load their configuration from `demo_strategies.yaml`:

```python
# Example from demo code (NOT hardcoded values!)
from rag.core.config_loader import ConfigLoader

# Load strategy configuration
config = ConfigLoader.from_strategy("research_papers_demo")

# Or from the YAML directly
config = ConfigLoader.from_file("demos/demo_strategies.yaml", strategy="research_papers_demo")

# The RAG system uses this configuration for ALL components
rag_system = RAGSystem(config)
```

To modify demo behavior:
1. Edit `demo_strategies.yaml` - change extractors, parsers, retrieval settings
2. No Python code changes needed - all behavior is configuration-driven
3. Add new extractors by listing them in the YAML configuration
4. Adjust search parameters (top_k, distance_metric) in the strategy config

## 🎓 Learning Outcomes

After running these demos, you'll understand:

1. **RAG Architecture Design**: How to structure modular, extensible RAG systems
2. **Domain Specialization**: Why and how to optimize for specific content types
3. **Extraction Strategies**: Different approaches for analyzing and enriching content
4. **Performance Optimization**: Techniques for efficient processing and retrieval
5. **Real-world Applications**: Practical use cases across multiple industries
6. **Configuration-First Development**: How to build systems where behavior is defined in YAML, not code

## 💡 Adding Sentiment Analysis

To add sentiment analysis to Demo 4 (or any demo), simply update `demo_strategies.yaml`:

```yaml
news_analysis_demo:
  components:
    extractors:
      - type: "EntityExtractor"
        config: { ... }
      - type: "SentimentExtractor"  # Add this!
        config:
          model: "local"  # or "api-based"
          granularity: "sentence"  # or "paragraph", "document"
          categories: ["positive", "negative", "neutral"]
          confidence_threshold: 0.7
      - type: "SummaryExtractor"
        config: { ... }
```

The framework's factory pattern automatically loads any configured extractor. No code changes needed!

## 🤝 Contributing

Want to add more demos or improve existing ones?

1. **Follow the demo structure**: Parser → Extractors → Embedder → Vector Store
2. **Use strategy configuration**: All settings in `demo_strategies.yaml`
3. **Create comprehensive samples**: Realistic data for your domain
4. **Add detailed analysis**: Show what makes your approach unique
5. **Include educational content**: Explain why this strategy works
6. **Test thoroughly**: Ensure demos run reliably
7. **No hardcoding**: Keep all configuration in YAML files

## 📝 License

These demos are part of the RAG system project and follow the same licensing terms.