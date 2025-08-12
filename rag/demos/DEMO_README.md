# RAG System Demos

This directory contains demonstration scripts showcasing the RAG system's capabilities using the strategy-based configuration approach.

## Quick Start

### Run All Demos
```bash
# Interactive menu to run all demos or select specific ones
python demos/run_all_cli_demos.py

# Or using uv
uv run python demos/run_all_cli_demos.py
```

### Run Individual Demos

#### CLI-Based Demos (NEW)
These demos showcase the platform through CLI commands only:

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

#### Enhanced Demos (with rich visualizations)
```bash
# Research Papers with enhanced output
python demos/demo1_research_papers_cli_enhanced.py
```

## Demo Overview

### Demo 1: Research Papers Analysis
- **Strategy**: `research_papers_demo`
- **Features**: Directory parsing, PDF/TXT/MD files, academic search
- **Highlights**: Shows verbose embeddings, metadata extraction, entity recognition

### Demo 2: Customer Support System
- **Strategy**: `customer_support_demo`
- **Features**: CSV & TXT parsing, ticket analysis, priority detection
- **Highlights**: Metadata filtering, sentiment analysis, knowledge base integration

### Demo 3: Code Documentation Analysis
- **Strategy**: `code_documentation_demo`
- **Features**: Markdown/Python parsing, code extraction, technical search
- **Highlights**: HybridUniversalStrategy, code pattern extraction

### Demo 4: News Article Analysis
- **Strategy**: `news_analysis_demo`
- **Features**: HTML/TXT parsing, entity extraction, temporal analysis
- **Highlights**: RerankedStrategy with recency boost, entity recognition

### Demo 5: Business Reports Analysis
- **Strategy**: `business_reports_demo`
- **Features**: PDF parsing, financial metrics extraction
- **Highlights**: Pattern extraction for financial data, MultiQueryStrategy

### Demo 6: Document Management & Vector DB
- **Features**: Add/search/delete/replace operations, metadata queries
- **Highlights**: Document lifecycle, deduplication, hashing, collection management
- **Note**: Uses multiple strategies to demonstrate different operations

## CLI Commands Reference

### Basic Commands
```bash
# Initialize a collection
python cli.py --strategy-file demos/demo_strategies.yaml init --strategy <strategy_name>

# Ingest documents
python cli.py --strategy-file demos/demo_strategies.yaml ingest --strategy <strategy_name> <path>

# Search with verbose output
python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy <strategy_name> "query" --top-k 5

# Get collection info
python cli.py --strategy-file demos/demo_strategies.yaml info --strategy <strategy_name>

# Delete collection
python cli.py --strategy-file demos/demo_strategies.yaml delete --strategy <strategy_name>

# Test system
python cli.py test
```

### Verbose Mode Features
When using `--verbose` flag, you'll see:
- 🧠 **Embedding visualizations**: Visual representation of vector embeddings
- 📊 **Processing progress**: Real-time document processing status
- 🏷️ **Metadata display**: Full metadata for each document
- 🔍 **Search details**: Similarity scores and ranking information
- ⚡ **Performance metrics**: Processing times and statistics

### Quiet Mode
Use `--quiet` for minimal output, useful for scripting:
```bash
python cli.py --strategy-file demos/demo_strategies.yaml --quiet search --strategy demo "query"
```

## Strategy Configuration

All demos use strategies defined in `demo_strategies.yaml`:

### Strategy Structure
```yaml
strategies:
  - name: "strategy_name"
    description: "Strategy description"
    components:
      parser:
        type: "DirectoryParser"  # Or specific parser
        config: {...}
      extractors:
        - type: "EntityExtractor"
          priority: 95
          config: {...}
      embedder:
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
      vector_store:
        type: "ChromaStore"
        config:
          collection_name: "collection"
      retrieval_strategy:
        type: "HybridUniversalStrategy"
        config: {...}
```

## Key Features Demonstrated

### 1. Universal Retrieval Strategies
- **BasicSimilarityStrategy**: Simple vector similarity
- **MetadataFilteredStrategy**: Intelligent metadata filtering
- **MultiQueryStrategy**: Query expansion and variations
- **RerankedStrategy**: Multi-factor re-ranking
- **HybridUniversalStrategy**: Combines multiple strategies

### 2. Advanced Parsing
- **DirectoryParser**: Handles entire directories with mixed file types
- **CustomerSupportCSVParser**: Specialized CSV handling
- **MarkdownParser**: Preserves structure and formatting
- **PlainTextParser**: Universal text handling

### 3. Rich Metadata Extraction
- Entity recognition (people, organizations, locations)
- Pattern extraction (emails, URLs, phone numbers)
- Summary generation
- Sentiment analysis
- Custom metadata fields

### 4. Vector Database Operations
- Collection management (create, delete, info)
- Document lifecycle (add, update, delete)
- Metadata-based filtering
- Deduplication via hashing
- Batch operations

## Sample Data

Each demo includes sample data in `static_samples/`:
- `research_papers/`: Academic papers and abstracts
- `customer_support/`: Support tickets and knowledge base
- `code_documentation/`: API docs and code samples
- `news_articles/`: News articles with entities
- `business_reports/`: Financial reports and analyses

## Troubleshooting

### Common Issues

1. **"No documents processed"**
   - Check if Ollama is running: `ollama list`
   - Ensure the embedding model is available: `ollama pull nomic-embed-text`

2. **"Strategy not found"**
   - Verify strategy exists in `demo_strategies.yaml`
   - Check for typos in strategy name

3. **"Collection already exists"**
   - Use `python cli.py delete --strategy <name>` to clear
   - Or check `demos/vectordb/` directory

4. **Slow embedding generation**
   - First run downloads the model
   - Subsequent runs use cached model
   - Consider using `--quiet` mode for faster execution

## Development

### Adding a New Demo

1. Create demo script in `demos/`
2. Add strategy to `demo_strategies.yaml`
3. Add sample data to `static_samples/`
4. Update this README
5. Add to `run_all_cli_demos.py`

### Testing Demos
```bash
# Test all demos quickly
python demos/run_all_cli_demos.py

# Test with verbose output
python cli.py --verbose test
```

## Performance Tips

1. **Use appropriate chunk sizes**: Smaller for precise search, larger for context
2. **Enable caching**: Strategies cache embeddings automatically
3. **Batch operations**: Ingest multiple files at once
4. **Use quiet mode**: For production and scripting
5. **Monitor memory**: Check collection sizes with `info` command

## Next Steps

After running the demos, try:

1. **Modify strategies**: Edit `demo_strategies.yaml` to change behavior
2. **Add your data**: Replace sample data with your documents
3. **Create custom strategies**: Combine different components
4. **Build applications**: Use the CLI or API in your apps
5. **Explore the API**: Import and use the RAG system programmatically

For more information, see the main [README.md](../README.md) in the parent directory.