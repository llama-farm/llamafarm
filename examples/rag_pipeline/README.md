# RAG Pipeline - Complete Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Dataset Management](#dataset-management)
- [RAG Operations](#rag-operations)
- [Using RAG with LLM](#using-rag-with-llm)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The RAG (Retrieval-Augmented Generation) pipeline in LlamaFarm allows you to enhance your LLM responses with relevant context from your document collections. This guide covers everything from initial setup to advanced querying.

### Key Features
- 🔍 **Intelligent Document Processing**: Automatically handles PDFs, Word docs, CSVs, Markdown, and more
- 🧠 **Vector Search**: Fast semantic search using ChromaDB and Ollama embeddings
- 🤖 **LLM Integration**: Seamlessly augments LLM responses with retrieved context
- ⚙️ **Flexible Configuration**: Config-driven with optional CLI overrides

## Quick Start

### Prerequisites

1. **Install Ollama and pull required models:**
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh

   # Start Ollama
   ollama serve

   # Pull embedding model
   ollama pull nomic-embed-text

   # Pull LLM model (if not already available)
   ollama pull llama3.1:8b
   ```

2. **Start LlamaFarm server:**
   ```bash
   # Using nx (recommended)
   nx start server

   # Or using lf directly
   lf server start
   ```

### Initialize Your Project

```bash
# Create a new LlamaFarm project
lf init my-rag-project

# Navigate to your project
cd my-rag-project
```

## Configuration

### Complete `llamafarm.yaml` Example

Create or update your `llamafarm.yaml` with this complete configuration:

```yaml
version: v1
name: my-rag-project
namespace: default

# System prompts for RAG-aware responses
prompts:
  - name: default
    messages:
      - role: system
        content: |
          You are a helpful AI assistant with access to a knowledge base through RAG (Retrieval-Augmented Generation).
          When context is provided from the knowledge base, use it to provide accurate and detailed answers.
          If relevant context is available, cite or reference it in your response.
          If no relevant context is provided, answer based on your general knowledge while noting that you don't have specific information from the knowledge base.

# Runtime configuration
runtime:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434/v1  # Note: /v1 for OpenAI compatibility
  temperature: 0.7

# RAG Configuration
rag:
  databases:
    - name: main_database
      type: ChromaStore
      config:
        collection_name: documents
        distance_function: cosine
        persist_directory: ./data/chroma_db
        port: 8000

      # Embedding strategies
      embedding_strategies:
        - name: default_embeddings
          type: OllamaEmbedder
          priority: 0
          config:
            auto_pull: true
            base_url: http://localhost:11434/
            batch_size: 16
            dimension: 768
            model: nomic-embed-text
            timeout: 60

      # Retrieval strategies
      retrieval_strategies:
        - name: basic_search
          type: BasicSimilarityStrategy
          default: true
          config:
            distance_metric: cosine
            top_k: 10

        - name: filtered_search
          type: MetadataFilteredStrategy
          default: false
          config:
            fallback_multiplier: 2
            filter_mode: post
            top_k: 10

      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: basic_search

  # Data processing strategy with all parsers and extractors
  data_processing_strategies:
    - name: universal_processor
      description: "Single strategy handling all document types with pattern-based routing"
      parsers:
        # PDF Parsers (with fallback)
        - type: PDFParser_LlamaIndex
          file_include_patterns: ["*.pdf", "*.PDF"]
          priority: 100
          config:
            chunk_strategy: semantic
            chunk_size: 1000
            chunk_overlap: 200
            extract_metadata: true
            extract_images: false
            preserve_equations: true

        - type: PDFParser_PyPDF2
          file_include_patterns: ["*.pdf", "*.PDF"]
          priority: 50
          config:
            chunk_strategy: paragraphs
            chunk_size: 1000
            chunk_overlap: 150
            extract_metadata: true

        # Word Document Parser
        - type: DocxParser_LlamaIndex
          file_include_patterns: ["*.docx", "*.DOCX", "*.doc", "*.DOC"]
          priority: 100
          config:
            chunk_size: 1000
            chunk_overlap: 150
            extract_metadata: true

        # CSV/TSV Parser
        - type: CSVParser_Pandas
          file_include_patterns: ["*.csv", "*.CSV", "*.tsv", "*.TSV", "*.dat"]
          priority: 100
          config:
            chunk_size: 500
            chunk_strategy: rows
            extract_metadata: true

        # Excel Parser
        - type: ExcelParser_LlamaIndex
          file_include_patterns: ["*.xlsx", "*.XLSX", "*.xls", "*.XLS"]
          priority: 100
          config:
            chunk_size: 500
            extract_metadata: true

        # Markdown Parser
        - type: MarkdownParser_Python
          file_include_patterns: ["*.md", "*.markdown", "*.mdown", "*.mkd", "README*", "CHANGELOG*"]
          priority: 100
          config:
            chunk_size: 1000
            chunk_strategy: sections
            extract_metadata: true
            extract_code_blocks: true
            extract_links: true

        # Universal Text Parser (catch-all)
        - type: TextParser_Python
          file_include_patterns: [
            "*.txt", "*.text", "*.log",
            "*.json", "*.xml", "*.yaml", "*.yml", "*.toml",
            "*.cfg", "*.conf", "*.ini", "*.properties",
            "*.py", "*.js", "*.java", "*.cpp", "*.c", "*.h", "*.go", "*.rs",
            "*.sh", "*.bash", "*.zsh", "*.ps1", "*.bat",
            "LICENSE*", "AUTHORS*", "NOTICE*", "COPYRIGHT*",
            "*.sql", "*.css", "*.html", "*.htm"
          ]
          priority: 10  # Lowest priority - catch-all
          config:
            chunk_size: 1000
            chunk_overlap: 100
            chunk_strategy: sentences
            extract_metadata: true
            encoding: utf-8
            clean_text: true

      # Extractors with pattern matching
      extractors:
        # Universal extractors (apply to all files)
        - type: ContentStatisticsExtractor
          file_include_patterns: ["*"]
          priority: 100
          config:
            include_readability: true
            include_structure: true
            include_vocabulary: true

        - type: EntityExtractor
          file_include_patterns: ["*"]
          priority: 90
          config:
            entity_types: [PERSON, ORG, GPE, DATE, PRODUCT, MONEY, PERCENT]
            min_entity_length: 2
            use_fallback: true

        - type: KeywordExtractor
          file_include_patterns: ["*"]
          priority: 80
          config:
            algorithm: yake
            max_keywords: 10
            min_keyword_length: 3

# Datasets (will be populated as you add data)
datasets: []
```

### Run the test script on this project
From the `examples/rag_pipeline` folder, run the following commands:
```bash
LLAMAFARM_CONFIG=~/REPLACE/WITH/PATH/TO/YOUR/PROJECT/llamafarm.yaml ./test_rag_comprehensive.sh
```

## Dataset Management

### Complete CLI Commands for Dataset Operations

```bash
# === CREATING DATASETS ===

# Create a new dataset
lf datasets add my-documents \
  --data-processing-strategy universal_processor \
  --database main_database

# Create dataset with description
lf datasets add research-papers \
  --data-processing-strategy universal_processor \
  --database main_database \
  --description "Academic papers and research documents"

# === LISTING AND VIEWING ===

# List all datasets
lf datasets list

# List datasets with details (JSON format)
lf datasets list --json

# Show detailed info about a specific dataset
lf datasets show my-documents

# List files in a dataset
lf datasets list-files my-documents

# === ADDING FILES ===

# Add a single file
lf datasets add-file my-documents /path/to/document.pdf

# Add multiple files
lf datasets add-file my-documents /path/to/file1.pdf /path/to/file2.md /path/to/file3.txt

# Add all PDFs from a directory
lf datasets add-file my-documents /documents/*.pdf

# Add files recursively from directory
lf datasets add-file my-documents /documents/**/*.pdf

# === REMOVING FILES AND DATASETS ===

# Remove a specific file from dataset (use file hash from list-files)
lf datasets remove-file my-documents 1d58f9207b989d7a34100b8e4f998ab1be9f5402b7cef62103a97c6d0dec1a06

# Remove multiple files
lf datasets remove-file my-documents hash1 hash2 hash3

# Delete entire dataset
lf datasets remove my-documents

# Force delete without confirmation
lf datasets remove my-documents --force
```

## RAG Operations

### Processing and Ingesting Documents

```bash
# === PROCESSING DOCUMENTS ===

# Process all files in a dataset (ingest into vector database)
lf rag process my-documents

# Process with specific strategy
lf rag process my-documents --strategy universal_processor

# Force reprocessing (even if already processed)
lf rag process my-documents --force

# Process with custom parameters
lf rag process my-documents \
  --chunk-size 500 \
  --chunk-overlap 50

# === SEARCHING THE VECTOR DATABASE ===

# Basic semantic search
lf rag search --database main_database "What is transformer architecture?"

# Search with custom number of results
lf rag search --database main_database "explain attention mechanism" --top-k 5

# Search with relevance threshold
lf rag search --database main_database "neural networks" --score-threshold 0.7

# Search with specific retrieval strategy
lf rag search --database main_database "BERT model" --retrieval-strategy filtered_search

# Combined search parameters
lf rag search --database main_database "deep learning concepts" \
  --top-k 10 \
  --score-threshold 0.6 \
  --retrieval-strategy basic_search

# === DATABASE STATISTICS ===

# Show statistics for a specific database
lf rag stats --database main_database

# Show statistics for all databases
lf rag stats

# Detailed statistics with metadata
lf rag stats --database main_database --detailed

# === DATABASE MANAGEMENT ===

# Clear a database (remove all vectors)
lf rag clear --database main_database

# Clear with confirmation
lf rag clear --database main_database --confirm

# Export database to file
lf rag export --database main_database ./backups/my-docs-backup.json

# Export with metadata
lf rag export --database main_database ./backups/my-docs-full.json --include-metadata

# Import database from file
lf rag import --database main_database ./backups/my-docs-backup.json

# Import with merge (don't overwrite existing)
lf rag import --database main_database ./backups/my-docs-backup.json --merge
```

## Using RAG with LLM

### Complete Examples of `lf chat` Commands

```bash
# === BASIC USAGE ===

# Simple query WITHOUT RAG (uses only LLM general knowledge)
lf chat "What is 2+2?"

# Simple query WITH automatic RAG (if configured in llamafarm.yaml)
lf chat "What is transformer architecture?"
# Note: RAG is auto-enabled if datasets exist and RAG is configured

# === EXPLICIT RAG CONTROL ===

# Explicitly enable RAG
lf chat --rag "What is transformer architecture?"

# Disable RAG even if configured (use --no-rag flag if implemented)
lf chat --no-rag "What is transformer architecture?"

# === RAG WITH SPECIFIC DATABASE ===

# Use a specific database
lf chat --rag --database main_database "Explain attention mechanism"

# Use alternative database
lf chat --rag --database technical_docs "API authentication methods"

# === CONTROLLING RETRIEVAL PARAMETERS ===

# Control number of retrieved documents
lf chat --rag --rag-top-k 5 "How do neural networks work?"

# Retrieve more context for complex queries
lf chat --rag --rag-top-k 20 "Summarize all security features"

# Set minimum relevance threshold
lf chat --rag --rag-score-threshold 0.8 "What is BERT?"

# Only get highly relevant results
lf chat --rag --rag-score-threshold 0.9 "Specific API endpoint details"

# === COMBINING ALL PARAMETERS ===

# Full control over RAG parameters
lf chat --rag \
  --database main_database \
  --rag-top-k 10 \
  --rag-score-threshold 0.7 \
  --retrieval-strategy filtered_search \
  "Explain the complete authentication flow"

# Complex query with specific requirements
lf chat --rag \
  --database research_papers \
  --rag-top-k 15 \
  --rag-score-threshold 0.6 \
  "Compare transformer and LSTM architectures"

# === DEBUGGING AND TESTING ===

# Run with debug output
lf chat --debug --rag "test query"

# Run with verbose output
lf chat --verbose --rag "What are the main features?"

# Test RAG without streaming
lf chat --no-stream --rag "Quick test"
```

### Comparing RAG vs Non-RAG Responses

```bash
# Example 1: Generic question
echo "=== Without RAG ==="
lf chat "What is machine learning?"
# Output: Generic textbook definition

echo "=== With RAG (your documents) ==="
lf chat --rag "What is machine learning?"
# Output: Definition enriched with specifics from your documents

# Example 2: Specific to your content
echo "=== Without RAG ==="
lf chat "What are our API rate limits?"
# Output: "I don't have specific information about your API rate limits..."

echo "=== With RAG ==="
lf chat --rag "What are our API rate limits?"
# Output: "Based on the documentation, the API rate limits are:
#         - Standard tier: 1000 requests/hour
#         - Premium tier: 10000 requests/hour..."
```

## Complete Workflow Examples

### Example 1: Setting Up Documentation Assistant

```bash
#!/bin/bash
# setup_docs_assistant.sh

echo "Setting up Documentation Assistant..."

# 1. Initialize project
lf init docs-assistant
cd ~/.llamafarm/projects/default/docs-assistant

# 2. Copy configuration
cat > llamafarm.yaml << 'EOF'
version: v1
name: docs-assistant
namespace: default
prompts:
  - name: default
    messages:
      - role: system
        content: |
          You are a technical documentation assistant.
          Use the provided context to give accurate, detailed answers.
          Always cite the source document when possible.
runtime:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434/v1
# ... (rest of config from above)
EOF

# 3. Create datasets for different doc types
lf datasets add api-docs --database main_database
lf datasets add user-guides --database main_database
lf datasets add troubleshooting --database main_database

# 4. Add documents
echo "Adding API documentation..."
lf datasets add-file api-docs ~/docs/api/*.md

echo "Adding user guides..."
lf datasets add-file user-guides ~/docs/guides/*.pdf

echo "Adding troubleshooting docs..."
lf datasets add-file troubleshooting ~/docs/troubleshooting/*.md

# 5. Process all datasets
for dataset in api-docs user-guides troubleshooting; do
  echo "Processing $dataset..."
  lf rag process $dataset
done

# 6. Verify setup
echo "Verification:"
lf datasets list
lf rag stats

# 7. Test queries
echo "Testing RAG queries..."
lf chat --rag "How do I authenticate with the API?"
lf chat --rag "What are the system requirements?"
lf chat --rag "How do I troubleshoot connection errors?"
```

### Example 2: Research Paper Analysis

```bash
#!/bin/bash
# research_assistant.sh

# Setup for analyzing research papers
echo "Setting up Research Paper Assistant..."

# 1. Create specialized dataset
lf datasets add research-papers \
  --database main_database \
  --description "ML/AI research papers"

# 2. Add papers
lf datasets add-file research-papers ~/papers/*.pdf

# 3. Process with semantic chunking
lf rag process research-papers

# 4. Analysis queries
queries=(
  "What are the key innovations in transformer architecture?"
  "Compare attention mechanisms across different papers"
  "Summarize findings on model scaling laws"
  "What are the main challenges mentioned in these papers?"
)

for query in "${queries[@]}"; do
  echo "Query: $query"
  lf chat --rag --rag-top-k 10 "$query"
  echo "---"
  sleep 2
done
```

### Example 3: Code Documentation Search

```bash
#!/bin/bash
# code_search.sh

# Setup for code and documentation search
echo "Setting up Code Documentation Search..."

# 1. Create dataset for code
lf datasets add codebase \
  --database main_database \
  --description "Source code and documentation"

# 2. Add various code files
lf datasets add-file codebase ~/project/src/**/*.py
lf datasets add-file codebase ~/project/src/**/*.js
lf datasets add-file codebase ~/project/docs/**/*.md

# 3. Process
lf rag process codebase

# 4. Code-specific queries
lf chat --rag "Find all authentication functions"
lf chat --rag "What database models are defined?"
lf chat --rag "Show me the API endpoint implementations"
lf chat --rag --rag-top-k 20 "List all error handling patterns"
```

## Advanced Configuration

### Multiple Databases for Different Domains

```yaml
rag:
  databases:
    # Technical documentation
    - name: technical_docs
      type: ChromaStore
      config:
        collection_name: technical
        persist_directory: ./data/technical_db
      embedding_strategies:
        - name: tech_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
            dimension: 768
      retrieval_strategies:
        - name: tech_search
          type: BasicSimilarityStrategy
          default: true
          config:
            top_k: 10

    # Customer data
    - name: customer_data
      type: ChromaStore
      config:
        collection_name: customers
        persist_directory: ./data/customer_db
      embedding_strategies:
        - name: customer_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
            dimension: 768
      retrieval_strategies:
        - name: customer_search
          type: MetadataFilteredStrategy
          default: true
          config:
            top_k: 5
            filter_mode: pre
```

### Custom Processing Strategies

```yaml
data_processing_strategies:
  # For source code
  - name: code_processor
    description: "Optimized for source code files"
    parsers:
      - type: TextParser_Python
        file_include_patterns: ["*.py", "*.js", "*.java", "*.go"]
        config:
          chunk_strategy: functions
          chunk_size: 500
          extract_metadata: true
          preserve_indentation: true
    extractors:
      - type: PatternExtractor
        config:
          patterns: ["function", "class", "import", "TODO", "FIXME"]

  # For research papers
  - name: research_processor
    description: "For academic papers"
    parsers:
      - type: PDFParser_LlamaIndex
        file_include_patterns: ["*.pdf"]
        config:
          chunk_strategy: semantic
          chunk_size: 1500
          preserve_equations: true
          extract_citations: true
    extractors:
      - type: EntityExtractor
        config:
          entity_types: [PERSON, ORG, DATE]
      - type: KeywordExtractor
        config:
          algorithm: yake
          max_keywords: 20
```

## Troubleshooting

### Common Issues and Solutions

#### "No response received" from `lf chat`
```bash
# Clear session context
rm -f ~/.llamafarm/session_context.yaml

# Try again
lf chat "test query"
```

#### RAG not finding relevant content
```bash
# Check if documents were processed
lf rag stats my-dataset

# If count is 0, reprocess
lf rag process my-dataset --force

# Try with lower threshold
lf chat --rag --rag-score-threshold 0.5 "your query"
```

#### Slow processing
```bash
# Use larger batch sizes in config
# batch_size: 32  # Instead of 16

# Process in parallel if possible
lf rag process my-dataset --workers 4
```

#### Server connection issues
```bash
# Check server status
curl http://localhost:8000/health

# Restart server
pkill -f "uvicorn.*main:app"
nx start server
```

### Debug Mode

```bash
# Enable debug output
lf chat --debug --rag "test query"

# Check debug log
tail -f ~/.llamafarm/projects/default/my-project/debug.log

# Verbose server logs
tail -f ~/logs/llamafarm-server.log
```

## Performance Optimization

1. **Chunk Size**:
   - Smaller (500-1000) for precise retrieval
   - Larger (2000-3000) for more context

2. **Batch Processing**:
   ```bash
   # Process multiple files together
   lf datasets add-file my-docs file1.pdf file2.pdf file3.pdf
   lf rag process my-docs
   ```

3. **Embedding Cache**: Automatically caches embeddings for faster repeated queries

4. **Metadata Filtering**: Use metadata to narrow search scope
   ```bash
   lf chat --rag --rag-filter "type:api" "authentication methods"
   ```

## What's Included in This Example

### Sample Documents
The `sample_files/` directory contains realistic example documents:

- **`research_papers/`** - Technical papers on AI/ML topics (transformer architecture, scaling laws)
- **`code_documentation/`** - API references and implementation guides
- **`news_articles/`** - HTML articles on technology topics
- **`code/`** - Python code examples

### Working Examples
- `rag_example.py` - Complete Python example of RAG pipeline
- `test_rag_cli.sh` - Bash script testing all CLI commands
- `llamafarm.yaml` - Complete configuration example
- Multiple file type support

### Main Example Script
`rag_example.py` shows how to:
1. Load and process documents
2. Generate embeddings with Ollama
3. Store vectors in ChromaDB
4. Query the knowledge base

## Key Features

### Dynamic Component Loading
All components are loaded from configuration - no hardcoding:
```yaml
embedding_strategies:
  - name: ollama_embeddings
    type: OllamaEmbedder
    config:
      model: nomic-embed-text
      dimension: 768
```

### Multiple File Types
Automatically handles:
- Text files (`.txt`)
- Markdown (`.md`)
- HTML files (`.html`)
- Python code (`.py`)
- PDFs, CSVs, and more (with appropriate parsers)

### Real Components
This example uses production-ready components:
- **OllamaEmbedder** - Generate embeddings with local models
- **ChromaDB** - Persistent vector storage
- **Multiple Parsers** - Handle various file formats
- **Content Extractors** - Extract metadata and structure

## CLI Commands for Dataset Management

### 1. Build the CLI Tool
```bash
# Navigate to CLI directory and build
cd cli
go build -o lf main.go
cd ..

# Make it executable (optional)
chmod +x ./lf
```

### 2. Initialize LlamaFarm Project
```bash
# Initialize a new project (if not already done)
./lf init

# Check project status
./lf status
```

### 3. Create a New Dataset
```bash
# Create dataset with RAG strategy and database
./lf datasets add \
  --data-processing-strategy universal_processor \
  --database main_database \
  my-documents

# Create dataset for specific document types
./lf datasets add \
  --data-processing-strategy pdf_processing \
  --database main_database \
  pdf-collection

# Create dataset with custom strategy from config file
./lf datasets add \
  --strategy-file examples/rag_pipeline/llamafarm.yaml \
  --data-processing-strategy universal_processor \
  --database main_database \
  research-papers
```

### 4. Add Data to Dataset
```bash
# Ingest a single file
./lf datasets ingest my-documents path/to/document.pdf

# Ingest multiple files
./lf datasets ingest my-documents \
  examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt \
  examples/rag_pipeline/sample_files/research_papers/neural_scaling_laws.txt

# Ingest all files from sample directories
./lf datasets ingest my-documents \
  examples/rag_pipeline/sample_files/research_papers/*.txt \
  examples/rag_pipeline/sample_files/code_documentation/*.md \
  examples/rag_pipeline/sample_files/news_articles/*.html

# Real-world example with various file types
./lf datasets ingest research-papers \
  rag/demos/static_samples/research_papers/transformer_architecture.txt \
  rag/demos/static_samples/customer_support/support_tickets.csv \
  rag/demos/static_samples/code_documentation/api_reference.md \
  rag/demos/static_samples/747/ryanair-737-700-800-fcom-rev-30.pdf
```

### 5. List and View Datasets
```bash
# List all datasets with file counts
./lf datasets list

# Show specific dataset details
./lf datasets info my-documents

# View dataset statistics
./lf datasets stats my-documents
```

### 6. Query Documents (Future Feature)
```bash
# Basic query
./lf rag query --dataset my-documents "What is transformer architecture?"

# Query with options
./lf rag query --dataset my-documents \
  --top-k 5 \
  --score-threshold 0.7 \
  "explain attention mechanism"

# Query across multiple datasets
./lf rag query --dataset my-documents,research-papers \
  "neural network scaling laws"
```

### 7. Manage Datasets
```bash
# Delete a dataset
./lf datasets delete my-documents

# Export dataset metadata
./lf datasets export my-documents --output dataset-export.json

# Import dataset from export
./lf datasets import dataset-export.json
```

### 8. Chat with RAG Context (Future Feature)
```bash
# Chat using specific dataset
./lf chat --rag --dataset my-documents "What papers discuss neural scaling?"

# Chat with retrieval settings
./lf chat --rag --dataset research-papers \
  --rag-top-k 10 \
  --rag-score-threshold 0.5 \
  "Summarize the documentation"
```

## Complete Example Workflow

Here's a full example of creating and populating a dataset:

```bash
# 1. Build the CLI
cd cli && go build -o lf main.go && cd ..

# 2. Initialize project (if needed)
./lf init

# 3. Create a dataset for research papers
./lf datasets add \
  --data-processing-strategy universal_processor \
  --database main_database \
  ai-research

# 4. Add sample documents to the dataset
./lf datasets ingest ai-research \
  examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt \
  examples/rag_pipeline/sample_files/research_papers/neural_scaling_laws.txt

# 5. Verify the ingestion
./lf datasets list

# Output should show:
# NAME          DATA PROCESSING STRATEGY   DATABASE        FILE COUNT
# ----          ------------------------   --------        ----------
# ai-research   universal_processor        main_database   2

# 6. Add more documents of different types
./lf datasets ingest ai-research \
  examples/rag_pipeline/sample_files/code_documentation/api_reference.md \
  examples/rag_pipeline/sample_files/news_articles/ai_breakthrough.html

# 7. Check updated file count
./lf datasets list

# Output should show:
# NAME          DATA PROCESSING STRATEGY   DATABASE        FILE COUNT
# ----          ------------------------   --------        ----------
# ai-research   universal_processor        main_database   4
```

## Using Custom Configuration

To use a custom configuration file with specific parsers and settings:

```bash
# 1. Create dataset with custom config
./lf datasets add \
  --strategy-file examples/rag_pipeline/llamafarm.yaml \
  --data-processing-strategy universal_processor \
  --database main_database \
  custom-dataset

# 2. The configuration defines:
#    - Which parsers to use for different file types
#    - Embedding model settings (e.g., Ollama nomic-embed-text)
#    - Vector store configuration (e.g., ChromaDB)
#    - Chunk sizes and overlap settings

# 3. Ingest files - they'll be processed according to config
./lf datasets ingest custom-dataset your-documents/*.pdf
```

## Customization

### Add Your Documents
Simply place files in the appropriate `sample_files/` subdirectory:
- Research papers → `research_papers/`
- Documentation → `code_documentation/`
- Articles → `news_articles/`
- Code → `code/`

### Modify Configuration
Edit `llamafarm.yaml` to:
- Change embedding models
- Adjust chunk sizes
- Configure different vector stores
- Add custom extractors

## Troubleshooting

### Ollama Issues
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart if needed
ollama serve
```

### Missing Models
```bash
# Pull required model
ollama pull nomic-embed-text
```

### Reset Database
```bash
# Clear ChromaDB data
rm -rf ./data/chroma_db
```

## Architecture Overview

```
RAG Pipeline Flow:
1. Documents → Parser (by file type)
2. Parsed content → Chunking (configurable size)
3. Chunks → Embedder (Ollama)
4. Embeddings → Vector Store (ChromaDB)
5. Query → Retrieval → Context → Response
```

## Learn More

- [LlamaFarm Documentation](https://docs.llamafarm.com)
- [Ollama Models](https://ollama.ai/library)
- [ChromaDB Guide](https://docs.trychroma.com)
