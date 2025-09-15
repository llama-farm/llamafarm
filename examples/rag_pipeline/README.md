# RAG Pipeline Examples

This directory contains a clean, working example of the LlamaFarm RAG (Retrieval-Augmented Generation) pipeline.

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
   ```

2. **Start LlamaFarm server:**
   ```bash
   lf server start
   ```

### Run the Example

```bash
# Navigate to examples directory
cd examples/rag_pipeline

# Run the RAG pipeline example
python rag_example.py
```

## What's Included

### Sample Documents
The `sample_files/` directory contains realistic example documents:

- **`research_papers/`** - Technical papers on AI/ML topics (transformer architecture, scaling laws)
- **`code_documentation/`** - API references and implementation guides 
- **`news_articles/`** - HTML articles on technology topics
- **`code/`** - Python code examples

### Configuration
The `llamafarm.yaml` demonstrates a complete RAG setup with:
- Ollama embeddings (768-dimensional vectors)
- ChromaDB vector storage
- Dynamic component loading
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
./lf run --rag --dataset my-documents "What papers discuss neural scaling?"

# Chat with retrieval settings
./lf run --rag --dataset research-papers \
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