# LlamaFarm Examples

Clean, practical examples demonstrating LlamaFarm CLI capabilities.

## Available Examples

### 📚 RAG Pipeline (`rag_pipeline/`)
A complete example of the Retrieval-Augmented Generation pipeline showing:
- Document ingestion using CLI commands
- Embedding generation with Ollama
- Vector storage with ChromaDB
- Semantic search and retrieval
- RAG-augmented chat completions

**Quick Start:**
```bash
# Run the complete example
./examples/rag_pipeline/run_example.sh

# Run comprehensive RAG test with duplicate detection
./examples/rag_pipeline/test_rag_comprehensive.sh

# Or view the detailed guide
cat examples/rag_pipeline/README.md
```

**Sample Files Included:**
- Research papers: `examples/rag_pipeline/sample_files/research_papers/`
  - `transformer_architecture.txt` - Transformer model overview
  - `neural_scaling_laws.txt` - ML scaling laws research
  - `llm_scaling_laws.txt` - Large language model scaling
- Code documentation: `examples/rag_pipeline/sample_files/code_documentation/`
  - `api_reference.md` - API documentation example
  - `best_practices.md` - Development best practices
  - `implementation_guide.md` - Implementation guidelines
- Code examples: `examples/rag_pipeline/sample_files/code/`
  - `example.py` - Python code sample
- News articles: `examples/rag_pipeline/sample_files/news_articles/`
  - `ai_breakthrough.html` - AI news article
  - `climate_tech_report.html` - Technology report
  - `quantum_computing_milestone.html` - Quantum computing news

## Prerequisites

All examples require:
1. **LlamaFarm CLI built:**
   ```bash
   cd cli && go build -o ../lf main.go && cd ..
   ```

2. **Server running** (the CLI will auto-start it if needed):
   ```bash
   nx start server
   ```

3. **Ollama running:**
   ```bash
   ollama serve
   ```

4. **Required models pulled:**
   ```bash
   # Chat model (choose one)
   ollama pull llama3.1:8b    # Currently configured
   ollama pull qwen3:8b        # Alternative

   # Embedding model (required for RAG)
   ollama pull nomic-embed-text
   ```

## Using the CLI

The LlamaFarm CLI (`lf`) provides all functionality. Here are examples using real files from the project:

### Dataset Management
```bash
# Create a dataset with processing strategy and database
./lf datasets add my-dataset -s universal_processor -b main_database

# Add real example documents (PDFs, text files, markdown)
./lf datasets ingest my-dataset examples/rag_pipeline/sample_files/research_papers/*.txt

# Add more document types
./lf datasets ingest my-dataset \
  examples/rag_pipeline/sample_files/code_documentation/*.md \
  examples/rag_pipeline/sample_files/code/*.py

# Add PDF documents (FDA letters, research papers, etc.)
./lf datasets ingest my-dataset \
  examples/rag_pipeline/sample_files/fda/*.pdf

# Process all uploaded files into the vector database
./lf datasets process my-dataset

# Re-process to see duplicate detection (files show as SKIPPED)
./lf datasets process my-dataset
# Files already in database will show:
#   ⏭️ Status: SKIPPED
#   Reason: All chunks already exist in database

# List all datasets
./lf datasets list

# Remove dataset when done
./lf datasets remove my-dataset
```

### RAG Queries
```bash
# Direct query to a specific database (searches your documents)
./lf rag query --database main_database "What is transformer architecture?"

# Query with custom retrieval settings
./lf rag query --database main_database --top-k 10 "Explain attention mechanism"

# Query with score threshold for more relevant results  
./lf rag query --database main_database --score-threshold 0.7 "Best practices for API design"
```

### Chat with RAG Integration
```bash
# Chat WITH RAG (default - augments responses with your documents)
./lf chat "What is transformer architecture?"

# Chat with specific database
./lf chat --database main_database "Explain attention mechanism"

# Debug mode to see retrieval details
./lf chat --database main_database --debug "Explain neural scaling laws"

# Chat WITHOUT RAG (LLM only, no document retrieval)
./lf chat --no-rag "What is machine learning?"
```

### Working with Real Documents

The project includes comprehensive sample documents you can use immediately:

```bash
# 1. Create a test dataset
./lf datasets add test-docs -s universal_processor -b main_database

# 2. Ingest research papers
./lf datasets ingest test-docs \
  examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt \
  examples/rag_pipeline/sample_files/research_papers/neural_scaling_laws.txt

# 3. Add FDA PDF documents
./lf datasets ingest test-docs \
  examples/rag_pipeline/sample_files/fda/761248_2024_Orig1s000OtherActionLtrs.pdf \
  examples/rag_pipeline/sample_files/fda/761240_2023_Orig1s000OtherActionLtrs.pdf

# 4. Add code documentation
./lf datasets ingest test-docs \
  examples/rag_pipeline/sample_files/code_documentation/api_reference.md \
  examples/rag_pipeline/sample_files/code_documentation/implementation_guide.md

# 5. Process all files into vector database
./lf datasets process test-docs

# 6. Query your documents with RAG
./lf rag query --database main_database "What are the key components of transformer architecture?"
./lf rag query --database main_database "What FDA submissions are discussed?"
./lf rag query --database main_database "What are the API authentication methods?"

# 7. Chat with RAG-augmented responses (default behavior)
./lf chat --database main_database "Explain neural scaling laws"
./lf chat --database main_database "What is BLA 761248?"

# 8. Compare RAG vs no-RAG responses
echo "=== With RAG (default - from your documents) ==="
./lf chat --database main_database "What is the DataProcessor class?"

echo "=== Without RAG (LLM general knowledge only) ==="
./lf chat --no-rag "What is the DataProcessor class?"
```

## Complete Example Workflow

Here's a full end-to-end example using the provided sample files:

```bash
# Step 1: Build the CLI (if not already done)
cd cli && go build -o ../lf main.go && cd ..

# Step 2: Ensure server is running
nx start server  # Or the CLI will auto-start it

# Step 3: Create a dataset for mixed documents
./lf datasets add demo-dataset -s universal_processor -b main_database

# Step 4: Ingest text documents
./lf datasets ingest demo-dataset \
  examples/rag_pipeline/sample_files/research_papers/*.txt

# Step 5: Ingest PDF documents
./lf datasets ingest demo-dataset \
  examples/rag_pipeline/sample_files/fda/*.pdf

# Step 6: Add code and documentation
./lf datasets ingest demo-dataset \
  examples/rag_pipeline/sample_files/code_documentation/*.md \
  examples/rag_pipeline/sample_files/code/*.py

# Step 7: Process all files into vector database
./lf datasets process demo-dataset

# Step 8: Verify dataset status
./lf datasets list
# Should show demo-dataset with file count

# Step 9: Query the knowledge base
./lf rag query --database main_database "What is transformer architecture?"
./lf rag query --database main_database "What FDA BLA submissions are mentioned?"
./lf rag query --database main_database --top-k 10 "API authentication methods"

# Step 10: Chat with RAG augmentation (default behavior)
./lf chat --database main_database "Explain the self-attention mechanism"
./lf chat --database main_database "What is BLA 761248 about?"
./lf chat --database main_database "How does the DataProcessor class work?"

# Step 11: Test different retrieval strategies
./lf rag query --database main_database --score-threshold 0.8 "neural scaling laws"
./lf rag query --database main_database --top-k 3 "best practices"

# Step 12: Clean up (optional)
./lf datasets remove demo-dataset
```

## Testing and Validation

### Comprehensive RAG Test Script

The `test_rag_comprehensive.sh` script provides a complete test of the RAG pipeline with enhanced features:

```bash
# Run the comprehensive test
./examples/rag_pipeline/test_rag_comprehensive.sh
```

**What it tests:**
1. **Database Creation** - Automatically adds a test database to llamafarm.yaml
2. **Dataset Management** - Creates and manages a test dataset
3. **Document Ingestion** - Ingests multiple document types (TXT, MD, PY, PDF)
4. **Processing Pipeline** - Processes documents into vector database
5. **RAG Queries** - Tests various query parameters (top-k, score threshold)
6. **Chat Integration** - Tests chat with and without RAG augmentation
7. **Duplicate Detection** - Verifies that re-processing shows files as SKIPPED
8. **Cleanup** - Optional removal of test data

**Key Features:**
- **Duplicate Detection Display**: Files processed multiple times show as:
  ```
  ⏭️ [1] transformer_architecture.txt
      ├─ Status: SKIPPED
      ├─ Reason: All chunks already exist in database
      └─ Action: No new data added (file previously processed)
  ```
- **Detailed Processing Info**: Shows parsers, extractors, embedders used
- **Colored Output**: Clear visual indicators for different statuses
- **Automatic Configuration**: Updates llamafarm.yaml automatically
- **Non-Interactive Mode**: Can be run in CI/CD pipelines

### Other Validation Scripts

```bash
# Quick validation of examples
./examples/validate_examples.sh

# Test RAG integration
./test/test_rag_integration.sh

# Verify lf chat commands
./test/verify_lf_run.sh

# Complete CLI flow test
./test/test_complete_cli_flow.py
```

## Configuration

The main configuration file is located at:
- Project config: `~/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml`
- Template config: `config/templates/default.yaml`

Example configuration sections:
- Runtime (LLM settings): Lines 378-381
- RAG databases: Lines 18-63
- Data processing strategies: Lines 64-348
- Datasets: Lines 349-377

## Directory Structure

```
examples/
├── README.md                     # This file
├── validate_examples.sh          # Validation script
└── rag_pipeline/                # RAG pipeline example
    ├── README.md                # Detailed RAG guide (1000+ lines)
    ├── run_example.sh           # Automated example script
    ├── llamafarm.yaml           # Example configuration
    └── sample_files/            # Sample documents
        ├── research_papers/     # AI/ML papers
        ├── code_documentation/  # API docs, guides
        ├── code/               # Python examples
        └── news_articles/      # HTML articles
```

## Additional Resources

- **Detailed RAG Guide**: `examples/rag_pipeline/README.md` - Comprehensive 1000+ line guide
- **API Documentation**: See server API docs at `http://localhost:8000/docs`
- **Test Scripts**: `test/` directory contains various test scripts

## Support

- [Documentation](https://docs.llamafarm.com)
- [GitHub Issues](https://github.com/llamafarm/llamafarm/issues)
- [Discord Community](https://discord.gg/llamafarm)