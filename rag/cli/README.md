# RAG CLI Complete Guide

A comprehensive guide to the RAG command-line interface (CLI) for document processing, search, and retrieval operations.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Command Overview](#command-overview)
3. [Global Options](#global-options)
4. [Core Commands](#core-commands)
   - [Test](#test-command)
   - [Ingest](#ingest-command)
   - [Search](#search-command)
   - [Info](#info-command)
   - [Manage](#manage-command)
   - [Strategies](#strategies-command)
   - [Extractors](#extractors-command)
5. [Strategy System](#strategy-system)
6. [Advanced Usage](#advanced-usage)
7. [Examples & Workflows](#examples--workflows)
8. [Troubleshooting](#troubleshooting)

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

---

## Command Overview

The RAG CLI provides a comprehensive set of commands for document management:

```bash
python cli.py [global-options] <command> [command-options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `test` | Test system components and configuration |
| `ingest` | Add documents to the vector database |
| `search` | Search for documents using various strategies |
| `info` | Display collection and system information |
| `manage` | Manage documents and collections |
| `strategies` | List and inspect available strategies |
| `extractors` | List available extractors |

---

## Global Options

These options can be used with any command:

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to configuration file | `--config config.json` |
| `--strategy-file` | Path to strategies YAML file | `--strategy-file strategies.yaml` |
| `--strategy` | Name of strategy to use | `--strategy research_papers_demo` |
| `--base-dir` | Base directory for relative paths | `--base-dir /path/to/project` |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `--log-level DEBUG` |
| `--verbose` | Enable verbose output | `--verbose` |
| `--quiet` | Suppress non-essential output | `--quiet` |

### Examples

```bash
# Use custom strategy file with verbose output
python cli.py --strategy-file my_strategies.yaml --verbose ingest documents/

# Use specific strategy with debug logging
python cli.py --strategy research_papers_demo --log-level DEBUG search "query"

# Quiet mode for scripting
python cli.py --quiet ingest data.csv
```

---

## Core Commands

### Test Command

Test system components and verify configuration.

```bash
python cli.py test
```

**Output:**
- ✅ Configuration validation
- ✅ Ollama connectivity
- ✅ Vector store accessibility
- ✅ Component availability

### Ingest Command

Add documents to the vector database with automatic parsing and extraction.

```bash
python cli.py ingest [options] <path>
```

**Arguments:**
- `path`: File or directory to ingest

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--strategy` | Strategy to use for ingestion | Required |
| `--recursive` | Recursively process directories | False |
| `--file-pattern` | File pattern to match (glob) | `*` |
| `--batch-size` | Number of files to process at once | 10 |
| `--continue-on-error` | Continue if individual files fail | False |

**Examples:**

```bash
# Ingest single file
python cli.py ingest document.pdf --strategy research_papers_demo

# Ingest directory recursively
python cli.py ingest docs/ --strategy code_documentation_demo --recursive

# Ingest specific file types
python cli.py ingest data/ --strategy news_analysis_demo --file-pattern "*.html"

# Batch processing with error handling
python cli.py ingest large_dataset/ --strategy business_reports_demo \
    --batch-size 50 --continue-on-error
```

### Search Command

Search documents using various retrieval strategies.

```bash
python cli.py search [options] <query>
```

**Arguments:**
- `query`: Search query string

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--strategy` | Strategy to use for search | Required |
| `--top-k` | Number of results to return | 5 |
| `--threshold` | Minimum similarity score | 0.0 |
| `--format` | Output format (text, json, table) | text |
| `--show-metadata` | Display document metadata | False |
| `--show-content` | Display full content | False |

**Examples:**

```bash
# Basic search
python cli.py search "transformer architecture" --strategy research_papers_demo

# Get more results with metadata
python cli.py search "customer complaint" --strategy customer_support_demo \
    --top-k 10 --show-metadata

# JSON output for integration
python cli.py search "API authentication" --strategy code_documentation_demo \
    --format json > results.json

# Filtered search with threshold
python cli.py search "quarterly revenue" --strategy business_reports_demo \
    --threshold 0.7 --top-k 3
```

### Info Command

Display information about collections and system status.

```bash
python cli.py info [options]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--strategy` | Show info for specific strategy |
| `--detailed` | Show detailed statistics |
| `--format` | Output format (text, json) |

**Examples:**

```bash
# Basic collection info
python cli.py info --strategy research_papers_demo

# Detailed statistics
python cli.py info --strategy customer_support_demo --detailed

# JSON format for monitoring
python cli.py info --strategy news_analysis_demo --format json
```

**Output includes:**
- Collection name and size
- Document count
- Storage location
- Index statistics
- Last update time

### Manage Command

Comprehensive document and collection management.

```bash
python cli.py manage [options] <subcommand>
```

#### Subcommands

##### Delete
Remove documents from the collection.

```bash
python cli.py manage delete [options] --strategy <strategy>
```

**Options:**
| Option | Description |
|--------|-------------|
| `--all` | Delete ALL documents in collection |
| `--older-than DAYS` | Delete documents older than N days |
| `--doc-ids ID [ID...]` | Delete specific document IDs |
| `--document-hashes HASH [HASH...]` | Delete by document hash |
| `--source-paths PATH [PATH...]` | Delete by source file path |
| `--expired` | Delete expired documents |
| `--delete-strategy` | Deletion strategy (soft, hard, archive) |
| `--dry-run` | Preview what would be deleted |

**Examples:**

```bash
# Delete all documents (with confirmation)
python cli.py manage delete --all --strategy research_papers_demo

# Delete old documents
python cli.py manage delete --older-than 30 --strategy news_analysis_demo

# Delete specific documents
python cli.py manage delete --doc-ids doc1 doc2 --strategy customer_support_demo

# Dry run to preview
python cli.py manage delete --all --strategy business_reports_demo --dry-run

# Delete by source file
python cli.py manage delete --source-paths /path/to/file.pdf --strategy research_papers_demo
```

##### Stats
Show collection statistics.

```bash
python cli.py manage stats --strategy <strategy> [--detailed]
```

##### Cleanup
Perform maintenance operations.

```bash
python cli.py manage cleanup --strategy <strategy> [options]
```

**Options:**
- `--duplicates`: Remove duplicate documents
- `--expired`: Clean up expired documents
- `--old-versions KEEP`: Keep only N latest versions

### Strategies Command

List and inspect available strategies.

```bash
python cli.py strategies [subcommand]
```

#### Subcommands

##### List
Show all available strategies.

```bash
python cli.py strategies list [--strategy-file path/to/strategies.yaml]
```

##### Show
Display detailed strategy configuration.

```bash
python cli.py strategies show <strategy-name> [--strategy-file path/to/strategies.yaml]
```

**Examples:**

```bash
# List all strategies
python cli.py strategies list

# Show specific strategy details
python cli.py strategies show research_papers_demo

# Use custom strategy file
python cli.py strategies list --strategy-file custom_strategies.yaml
```

### Extractors Command

List available metadata extractors.

```bash
python cli.py extractors list
```

**Output:**
- Extractor name
- Description
- Supported file types
- Configuration options

---

## Strategy System

Strategies define complete processing pipelines including parsers, extractors, embedders, and stores.

### Using Strategies

```bash
# Specify strategy for any operation
python cli.py --strategy research_papers_demo ingest documents/
python cli.py --strategy research_papers_demo search "query"
```

### Strategy File Format

```yaml
strategy_name:
  description: "Strategy description"
  
  # Document parsers
  parsers:
    - type: PDFParser
      config:
        extract_images: false
    - type: TextParser
      config:
        encoding: utf-8
  
  # Metadata extractors
  extractors:
    - type: KeywordExtractor
      config:
        max_keywords: 10
    - type: EntityExtractor
      config:
        entities: ["PERSON", "ORG"]
  
  # Embedding configuration
  embedder:
    type: OllamaEmbedder
    config:
      model: nomic-embed-text
      dimension: 768
  
  # Vector store configuration
  vector_store:
    type: ChromaStore
    config:
      collection_name: my_collection
  
  # Retrieval configuration
  retrieval:
    type: RerankedStrategy
    config:
      initial_k: 20
      final_k: 5
```

### Built-in Strategies

The system includes several pre-configured strategies:

1. **research_papers_demo** - Academic papers with citations
2. **customer_support_demo** - Support tickets with priority
3. **code_documentation_demo** - Technical docs with code blocks
4. **news_analysis_demo** - News articles with entities
5. **business_reports_demo** - Financial documents with tables
6. **document_management_demo** - General document processing

---

## Advanced Usage

### Batch Processing

Process multiple queries or files efficiently:

```bash
# Batch ingest with progress
find documents/ -name "*.pdf" | \
  xargs -I {} python cli.py ingest {} --strategy research_papers_demo

# Batch search from file
cat queries.txt | while read query; do
  python cli.py search "$query" --strategy customer_support_demo
done
```

### Integration with Scripts

```bash
#!/bin/bash
# backup_and_reingest.sh

# Backup current collection
python cli.py info --strategy my_strategy --format json > backup_info.json

# Clean collection
python cli.py manage delete --all --strategy my_strategy

# Reingest with new configuration
python cli.py ingest /data/documents --strategy my_strategy --recursive

# Verify
python cli.py info --strategy my_strategy
```

### Monitoring and Logging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python cli.py ingest documents/ --strategy research_papers_demo

# Log to file
python cli.py search "query" --strategy news_analysis_demo 2> search.log

# Monitor collection growth
watch -n 60 'python cli.py info --strategy business_reports_demo'
```

### Pipeline Automation

```python
#!/usr/bin/env python
# automated_pipeline.py

import subprocess
import json

def run_cli_command(command):
    """Execute CLI command and return output."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout, result.returncode

# Ingest new documents
stdout, code = run_cli_command(
    "python cli.py ingest new_docs/ --strategy research_papers_demo"
)

if code == 0:
    # Search for specific content
    stdout, code = run_cli_command(
        'python cli.py search "machine learning" --strategy research_papers_demo --format json'
    )
    
    results = json.loads(stdout)
    print(f"Found {len(results)} relevant documents")
```

---

## Examples & Workflows

### Research Workflow

```bash
# 1. Setup collection
python cli.py manage delete --all --strategy research_papers_demo

# 2. Ingest papers
python cli.py ingest papers/ --strategy research_papers_demo --recursive

# 3. Verify ingestion
python cli.py info --strategy research_papers_demo

# 4. Search for topics
python cli.py search "transformer architecture" --strategy research_papers_demo --top-k 10

# 5. Export results
python cli.py search "attention mechanism" --strategy research_papers_demo \
    --format json > attention_papers.json
```

### Customer Support Workflow

```bash
# 1. Ingest support tickets
python cli.py ingest tickets.csv --strategy customer_support_demo

# 2. Search for similar issues
python cli.py search "password reset problem" --strategy customer_support_demo \
    --show-metadata --top-k 5

# 3. Analyze patterns
python cli.py manage stats --strategy customer_support_demo --detailed

# 4. Clean old tickets
python cli.py manage delete --older-than 90 --strategy customer_support_demo
```

### Documentation Management

```bash
# 1. Index documentation
python cli.py ingest docs/ --strategy code_documentation_demo \
    --recursive --file-pattern "*.md"

# 2. Search API docs
python cli.py search "authentication API" --strategy code_documentation_demo \
    --show-content

# 3. Update documentation
python cli.py manage delete --source-paths docs/old_api.md --strategy code_documentation_demo
python cli.py ingest docs/new_api.md --strategy code_documentation_demo

# 4. Verify updates
python cli.py info --strategy code_documentation_demo --detailed
```

---

## Troubleshooting

### Common Issues

#### Ollama Connection Error
```
Error: Failed to connect to Ollama at http://localhost:11434
```
**Solution:**
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

#### Collection Not Found
```
Error: Collection 'my_collection' does not exist
```
**Solution:**
```bash
# Create collection by ingesting first document
python cli.py ingest sample.txt --strategy my_strategy
```

#### Memory Issues with Large Files
```
Error: Out of memory while processing large file
```
**Solution:**
```bash
# Use smaller batch size
python cli.py ingest large_files/ --strategy my_strategy --batch-size 5

# Or process files individually
find large_files/ -type f -exec python cli.py ingest {} --strategy my_strategy \;
```

#### Duplicate Documents
```
Warning: Document already exists in collection
```
**Solution:**
The system automatically handles duplicates via content hashing. To force re-ingestion:
```bash
# Clean and re-ingest
python cli.py manage delete --source-paths file.pdf --strategy my_strategy
python cli.py ingest file.pdf --strategy my_strategy
```

### Debug Mode

Enable detailed debugging information:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or use command line
python cli.py --log-level DEBUG ingest documents/ --strategy my_strategy

# Save debug output
python cli.py --log-level DEBUG search "query" --strategy my_strategy 2> debug.log
```

### Performance Optimization

```bash
# Increase batch size for faster ingestion
python cli.py ingest large_dataset/ --strategy my_strategy --batch-size 100

# Use multiple workers (if supported)
export WORKERS=4
python cli.py ingest documents/ --strategy my_strategy

# Optimize search with caching
export ENABLE_CACHE=true
python cli.py search "common query" --strategy my_strategy
```

---

## Environment Variables

Configure the CLI behavior with environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage directory | `./chromadb` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `BATCH_SIZE` | Default batch size | `10` |
| `WORKERS` | Number of parallel workers | `1` |
| `ENABLE_CACHE` | Enable result caching | `false` |

---

## Command Reference Card

### Quick Reference

```bash
# System
python cli.py test                                    # Test setup
python cli.py strategies list                         # List strategies
python cli.py extractors list                         # List extractors

# Ingestion
python cli.py ingest <path> --strategy <name>         # Basic ingest
python cli.py ingest <dir> --strategy <name> --recursive  # Recursive

# Search
python cli.py search "query" --strategy <name>        # Basic search
python cli.py search "query" --strategy <name> --top-k 10  # More results

# Management
python cli.py info --strategy <name>                  # Collection info
python cli.py manage stats --strategy <name>          # Statistics
python cli.py manage delete --all --strategy <name>   # Clean collection

# Advanced
python cli.py --verbose <command>                     # Verbose output
python cli.py --quiet <command>                       # Quiet mode
python cli.py --log-level DEBUG <command>            # Debug mode
```

---

**Need more help?** Check the [main documentation](../README.md) or run `python cli.py <command> --help` for command-specific options.