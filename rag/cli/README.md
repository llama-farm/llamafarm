# RAG CLI Complete Guide

A comprehensive guide to the RAG command-line interface (CLI) for document processing, search, and retrieval operations.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Command Overview](#command-overview)
3. [Global Options](#global-options)
4. [Core Commands](#core-commands)
   - [Ingest](#ingest-command)
   - [Search](#search-command)
   - [Info](#info-command)
   - [Manage](#manage-command)
   - [Strategies](#strategies-command)
   - [Extractors](#extractors-command)
   - [Test](#test-command)
5. [Strategy System](#strategy-system)
6. [Advanced Usage](#advanced-usage)
7. [Examples & Workflows](#examples--workflows)
8. [Troubleshooting](#troubleshooting)
9. [Command Reference](#command-reference)

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running (for embeddings)
- Required Python packages (install via `pip install -r requirements.txt` or `uv sync`)

### Initial Setup

1. **Install Ollama embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

2. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. **Test the CLI:**
   ```bash
   python cli.py test
   ```

## Command Overview

The RAG CLI follows this general syntax pattern:

```bash
python cli.py [global-options] <command> [positional-args] [command-options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `ingest` | Process and index documents into vector database |
| `search` | Search indexed documents using semantic similarity |
| `info` | Display collection information and statistics |
| `manage` | Manage documents (delete, replace, stats, cleanup) |
| `strategies` | List, show, and manage RAG strategies |
| `extractors` | List available document extractors |
| `test` | Test system components and configuration |

## Global Options

Global options must come **before** the command:

| Option | Description | Example |
|--------|-------------|---------|
| `--config <path>` | Configuration file path | `--config myconfig.json` |
| `--base-dir <path>` | Base directory for operations | `--base-dir /data` |
| `--log-level <level>` | Logging level (DEBUG, INFO, WARNING, ERROR) | `--log-level DEBUG` |
| `--quiet` | Suppress non-essential output | `--quiet` |
| `--verbose` | Show detailed output | `--verbose` |
| `--content-length <n>` | Maximum content length in search results | `--content-length 500` |
| `--strategy-file <path>` | Custom strategy file path | `--strategy-file custom_strategies.yaml` |

## Core Commands

### Ingest Command

Process and index documents into the vector database.

#### Syntax
```bash
python cli.py ingest <path> [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--strategy <name>` | Use predefined strategy | None |
| `--parser <type>` | Override parser type | Auto-detect |
| `--embedder <type>` | Override embedder | From config |
| `--vector-store <type>` | Override vector store | From config |
| `--extractors <list>` | Extractors to use | From config |
| `--batch-size <n>` | Processing batch size | 10 |
| `--recursive` | Process directories recursively | False |
| `--file-filter <pattern>` | File pattern filter | None |

#### Examples
```bash
# Ingest with strategy
python cli.py ingest ./documents --strategy research_papers_demo

# Ingest specific file types recursively
python cli.py ingest ./data --recursive --file-filter "*.pdf"

# Custom configuration
python cli.py ingest ./docs --parser PDFParser --embedder OllamaEmbedder

# Use custom strategy file
python cli.py --strategy-file my_strategies.yaml ingest ./docs --strategy my_custom_strategy
```

### Search Command

Search indexed documents using semantic similarity.

#### Syntax
```bash
python cli.py search "query" [options]
```

#### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--strategy <name>` | Use predefined strategy | None |
| `--top-k <n>` | Number of results | 5 |
| `--retrieval <type>` | Retrieval strategy | From config |
| `--embedder <type>` | Override embedder | From config |
| `--vector-store <type>` | Override vector store | From config |

#### Examples
```bash
# Basic search
python cli.py search "machine learning algorithms"

# Search with strategy
python cli.py search "transformer architecture" --strategy research_papers_demo

# Verbose search with more results
python cli.py --verbose search "quantum computing" --top-k 10

# Quiet search with limited content
python cli.py --quiet --content-length 200 search "financial reports"
```

### Info Command

Display information about the vector database collection.

#### Syntax
```bash
python cli.py info [options]
```

#### Options
| Option | Description |
|--------|-------------|
| `--strategy <name>` | Use strategy configuration |
| `--vector-store <type>` | Override vector store |

#### Examples
```bash
# Get info using strategy
python cli.py info --strategy research_papers_demo

# Get info with specific vector store
python cli.py info --vector-store ChromaStore
```

### Manage Command

Manage documents in the collection.

#### Syntax
```bash
python cli.py manage [--rag-strategy <name>] <subcommand> [options]
```

#### Subcommands

##### stats
Display collection statistics.
```bash
python cli.py manage --rag-strategy research_papers_demo stats
```

##### delete
Delete documents from collection.
```bash
# Delete by document IDs
python cli.py manage delete --doc-ids doc1 doc2 doc3

# Delete documents older than N days
python cli.py manage delete --older-than 30

# Delete by filename
python cli.py manage delete --filenames "report.pdf" "data.csv"

# Soft delete (mark as deleted)
python cli.py manage delete --strategy soft --doc-ids doc1

# Hard delete (permanent removal)
python cli.py manage delete --strategy hard --doc-ids doc1
```

##### replace
Replace document content while preserving ID.
```bash
python cli.py manage replace --doc-id doc1 --new-content "Updated content"
```

##### cleanup
Clean up orphaned or duplicate documents.
```bash
python cli.py manage cleanup --remove-duplicates
```

##### hash
Generate or verify document hashes.
```bash
python cli.py manage hash --verify
```

### Strategies Command

Manage and explore RAG strategies.

#### Syntax
```bash
python cli.py strategies <subcommand> [options]
```

#### Subcommands

##### list
List all available strategies.
```bash
python cli.py strategies list
```

##### show
Show details of a specific strategy.
```bash
python cli.py strategies show research_papers_demo
```

##### recommend
Get strategy recommendations based on criteria.
```bash
python cli.py strategies recommend --use-case "academic research"
```

##### convert
Convert strategy to configuration file.
```bash
python cli.py strategies convert research_papers_demo --output config.yaml
```

##### test
Test a strategy configuration.
```bash
python cli.py strategies test research_papers_demo
```

### Extractors Command

Manage document extractors.

#### Syntax
```bash
python cli.py extractors <subcommand> [options]
```

#### Subcommands

##### list
List available extractors.
```bash
python cli.py extractors list
```

##### test
Test an extractor on a file.
```bash
python cli.py extractors test EntityExtractor --file sample.txt
```

### Test Command

Test system components and configuration.

#### Syntax
```bash
python cli.py test [options]
```

#### Options
| Option | Description |
|--------|-------------|
| `--test-file <path>` | Test parsing a specific file |

#### Examples
```bash
# Basic system test
python cli.py test

# Test file parsing
python cli.py test --test-file documents/sample.pdf
```

## Strategy System

Strategies are predefined configurations that optimize the RAG pipeline for specific use cases.

### Available Demo Strategies

| Strategy | Use Case | Key Features |
|----------|----------|--------------|
| `research_papers_demo` | Academic research | Citation extraction, statistical analysis |
| `customer_support_demo` | Support tickets | Priority detection, sentiment analysis |
| `code_documentation_demo` | Technical docs | Code extraction, API references |
| `news_analysis_demo` | News articles | Entity recognition, temporal analysis |
| `business_reports_demo` | Financial docs | Metrics extraction, table processing |

### Strategy Configuration

Strategies are defined in YAML files. By default, the CLI uses `default_strategies.yaml`, but you can specify custom strategy files:

```bash
# Use default strategy file
python cli.py search "query" --strategy research

# Use custom strategy file
python cli.py --strategy-file demos/demo_strategies.yaml search "query" --strategy research_papers_demo
```

Strategies are defined in YAML files:

```yaml
strategy_name:
  description: "Strategy description"
  use_cases: ["Use case 1", "Use case 2"]
  components:
    parser:
      type: "ParserType"
      config:
        setting1: value1
    extractors:
      - type: "ExtractorType"
        config:
          setting1: value1
    embedder:
      type: "EmbedderType"
      config:
        model: "model-name"
    vector_store:
      type: "VectorStoreType"
      config:
        collection_name: "collection"
    retrieval_strategy:
      type: "RetrievalType"
      config:
        top_k: 5
```

## Advanced Usage

### Batch Processing

Process multiple directories or files:

```bash
# Process multiple directories
for dir in data1 data2 data3; do
    python cli.py ingest $dir --strategy research_papers_demo
done

# Process files matching pattern
find . -name "*.pdf" -exec python cli.py ingest {} \;
```

### Pipeline Automation

Create automated pipelines:

```bash
#!/bin/bash
# Automated RAG pipeline

# 1. Clean previous data
python cli.py manage cleanup --remove-all

# 2. Ingest new documents
python cli.py ingest ./new_docs --strategy business_reports_demo

# 3. Generate statistics
python cli.py manage stats > stats.txt

# 4. Run test queries
python cli.py search "quarterly revenue" --top-k 10 > results.txt
```

### Custom Configuration Files

Create custom configuration files:

```json
{
  "parser": {
    "type": "PDFParser",
    "config": {
      "extract_images": true,
      "extract_metadata": true
    }
  },
  "embedder": {
    "type": "OllamaEmbedder",
    "config": {
      "model": "nomic-embed-text",
      "batch_size": 32
    }
  },
  "vector_store": {
    "type": "ChromaStore",
    "config": {
      "collection_name": "my_collection",
      "persist_directory": "./my_vectordb"
    }
  }
}
```

Use custom configuration:
```bash
python cli.py --config my_config.json ingest ./documents
```

## Examples & Workflows

### Academic Research Workflow

```bash
# 1. Initialize collection for research papers
python cli.py ingest ./papers --strategy research_papers_demo

# 2. Search for specific topics
python cli.py search "transformer architecture attention mechanism" --top-k 10

# 3. Get collection statistics
python cli.py info --strategy research_papers_demo

# 4. Export relevant papers
python cli.py search "neural networks 2023" --top-k 20 > relevant_papers.txt
```

### Customer Support Workflow

```bash
# 1. Ingest support tickets and knowledge base
python cli.py ingest ./tickets.csv --strategy customer_support_demo
python cli.py ingest ./knowledge_base.txt --strategy customer_support_demo

# 2. Search for similar issues
python cli.py search "login authentication failed" --top-k 5

# 3. Get high-priority tickets
python cli.py search "priority:critical" --strategy customer_support_demo

# 4. Clean up old tickets
python cli.py manage delete --older-than 90
```

### Document Management Workflow

```bash
# 1. Check collection status
python cli.py manage --rag-strategy document_management_demo stats

# 2. Remove duplicates
python cli.py manage cleanup --remove-duplicates

# 3. Update specific document
python cli.py manage replace --doc-id doc123 --new-content "Updated content"

# 4. Verify document integrity
python cli.py manage hash --verify
```

## Troubleshooting

### Common Issues and Solutions

#### "Strategy not found" Error
**Problem:** The specified strategy doesn't exist.
**Solution:** 
- List available strategies: `python cli.py strategies list`
- Check strategy name spelling
- Ensure strategy file is in correct location

#### "Ollama connection failed" Error
**Problem:** Cannot connect to Ollama embedding service.
**Solution:**
- Start Ollama: `ollama serve`
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Verify model is installed: `ollama list`

#### "No documents found" Error
**Problem:** Search returns no results.
**Solution:**
- Verify documents were ingested: `python cli.py info`
- Check collection name matches
- Try broader search terms
- Increase `--top-k` value

#### "Out of memory" Error
**Problem:** System runs out of memory during processing.
**Solution:**
- Reduce batch size: `--batch-size 5`
- Process files individually
- Use smaller embedding model
- Increase system swap space

#### Slow Performance
**Problem:** Operations take too long.
**Solution:**
- Use appropriate batch sizes
- Enable caching in configuration
- Use faster embedding models
- Process files in parallel

### Debug Mode

Enable debug logging for detailed information:

```bash
python cli.py --log-level DEBUG search "test query"
```

### Configuration Validation

Test configuration before use:

```bash
python cli.py test --config my_config.json
```

## Command Reference

### Quick Reference Table

| Command | Common Usage | Purpose |
|---------|-------------|---------|
| `ingest <path>` | `ingest ./docs --strategy demo` | Index documents |
| `search "query"` | `search "AI" --top-k 10` | Search documents |
| `info` | `info --strategy demo` | Show statistics |
| `manage stats` | `manage --rag-strategy demo stats` | Collection stats |
| `manage delete` | `manage delete --doc-ids id1` | Remove documents |
| `strategies list` | `strategies list` | List strategies |
| `extractors list` | `extractors list` | List extractors |
| `test` | `test --test-file doc.pdf` | Test system |

### Environment Variables

You can set these environment variables to customize behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_CONFIG_PATH` | Default config file path | `./config.json` |
| `RAG_BASE_DIR` | Default base directory | Current directory |
| `RAG_LOG_LEVEL` | Default log level | `INFO` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

### Exit Codes

The CLI uses standard exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Connection error |
| 5 | File not found |

## Best Practices

1. **Always use strategies** for consistent configuration
2. **Test configurations** before production use
3. **Monitor performance** with verbose mode during development
4. **Clean up regularly** to maintain optimal performance
5. **Backup vector databases** before major operations
6. **Use appropriate batch sizes** based on system resources
7. **Document custom strategies** for team collaboration

## Further Resources

- [Main README](../README.md) - Project overview and setup
- [Demo Guide](../demos/README.md) - Interactive demonstrations
- [API Documentation](./api-guide.md) - Programmatic usage
- [Strategy Guide](./strategy-guide.md) - Creating custom strategies

---

*Last updated: 2024*