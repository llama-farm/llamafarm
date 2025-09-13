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

## Common Commands

### Query Documents
```bash
# Basic query
lf rag query "What is transformer architecture?"

# Query with options
lf rag query --top-k 5 --score-threshold 0.7 "explain attention mechanism"
```

### Manage Database
```bash
# View statistics
lf rag stats

# List documents
lf rag list

# Clear database
lf rag clear --force
```

### Chat with RAG
```bash
# Chat with RAG context
lf run --rag "What papers discuss neural scaling?"

# Specify retrieval settings
lf run --rag --rag-top-k 10 "Summarize the documentation"
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