# RAG System Setup Guide

## 🚀 Quick Start Scripts

This RAG system includes several setup and demo scripts for easy deployment and testing:

### 1. Full System Setup and Demo
```bash
./setup_and_demo.sh
```
**What it does:**
- Installs all dependencies (uv, Ollama, Python packages)
- Sets up virtual environment
- Downloads embedding models
- Runs comprehensive demos of all features
- Shows usage examples

**Options:**
- `--skip-prompts`: Run without user prompts
- `--cleanup`: Clean up demo data on exit
- `--tests-only`: Only run system tests, skip demos

### 2. Quick Extractor Demo
```bash
./scripts/quick_extractor_demo.sh
```
**What it does:**
- Tests all 5 local-only extractors
- Shows extractor comparisons
- Demonstrates different text analysis capabilities
- No full system setup required

### 3. Simple Extractor Test
```bash
./scripts/test_extractors.sh
```
**What it does:**
- Quick non-interactive test of all extractors
- Verifies system functionality
- Returns pass/fail status for each extractor

## 📋 Manual Setup (Alternative)

If you prefer manual setup or need to customize the installation:

### Prerequisites
- macOS (scripts designed for macOS, but adaptable)
- Python 3.8+
- Internet connection for downloads

### Step-by-Step Manual Setup

1. **Install uv (Python package manager)**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

3. **Start Ollama and download model**
   ```bash
   # Start Ollama service (in background)
   nohup ollama serve > /dev/null 2>&1 &
   
   # Download embedding model
   ollama pull nomic-embed-text
   ```

4. **Setup Python environment**
   ```bash
   # Create virtual environment
   uv venv
   
   # Activate environment
   source .venv/bin/activate
   
   # Install dependencies
   uv pip install -e .
   
   # Install optional dependencies
   uv pip install python-dateutil textblob
   ```

5. **Test the installation**
   ```bash
   # Test extractor system
   uv run python cli.py extractors list
   
   # Test with sample data
   uv run python cli.py extractors test --extractor yake
   ```

## 🔧 System Features Demonstrated

The setup scripts demonstrate these key features:

### Document Processing
- CSV ingestion with customer support data
- PDF document processing
- Batch file processing
- File type auto-detection

### Extractors (Local-Only)
- **YAKE**: Advanced keyword extraction
- **RAKE**: Fast phrase extraction
- **TF-IDF**: Term frequency analysis
- **Entities**: Person, organization, date, email, phone extraction
- **DateTime**: Date and time extraction
- **Statistics**: Readability and content analysis

### Search Capabilities
- Basic similarity search
- Metadata-enhanced search
- Different retrieval strategies
- Query embedding and similarity scoring

### Document Management
- Document statistics and analytics
- Hash-based duplicate detection
- Soft/hard deletion strategies
- Cleanup and maintenance operations

## 📊 Sample Data Included

The system includes sample data for testing:

### CSV Data
- `samples/csv/small_sample.csv`: Customer support tickets
- `samples/csv/large_sample.csv`: Extended dataset
- `samples/csv/filtered-english-incident-tickets.csv`: Real-world data

### PDF Documents
- `samples/pdfs/llama.pdf`: Various PDF documents for testing
- `samples/pdfs/test_document.pdf`: Simple test document

## ⚙️ Configuration Examples

Pre-configured examples in `config_examples/`:

- `extractors_demo_config.json`: Extractor integration examples
- `unified_multi_strategy_config.json`: Multi-strategy configuration
- `enterprise_document_management_config.json`: Enterprise features
- `basic_config.json`: Simple starting configuration

## 🎯 Next Steps

After running the setup:

1. **Explore the extractors:**
   ```bash
   uv run python cli.py extractors list --detailed
   ```

2. **Try with your own data:**
   ```bash
   uv run python cli.py ingest your_file.csv --extractors yake entities
   ```

3. **Search your documents:**
   ```bash
   uv run python cli.py search "your search query"
   ```

4. **Customize configurations:**
   - Edit `config_examples/*.json` files
   - Create custom extractor configurations
   - Add new retrieval strategies

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not responding**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Restart if needed
   pkill ollama
   ollama serve
   ```

2. **Virtual environment issues**
   ```bash
   # Remove and recreate
   rm -rf .venv
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Import errors**
   ```bash
   # Make sure you're in the activated environment
   source .venv/bin/activate
   
   # Verify installation
   uv pip list
   ```

4. **ChromaDB permission issues**
   ```bash
   # Clean up ChromaDB data
   rm -rf ./data/*_chroma_db
   ```

### Getting Help

- Check the logs: Most CLI commands show detailed error messages
- Run with debug logging: Add `--log-level DEBUG` to commands
- Verify dependencies: Use `./scripts/test_extractors.sh` for quick verification
- Check configurations: Ensure JSON files are valid

## 🦙 Success!

If all scripts run successfully, you'll have a fully functional RAG system with:
- ✅ Local-only extractors for privacy-safe processing
- ✅ Multiple document formats supported
- ✅ Flexible configuration system
- ✅ Comprehensive search capabilities
- ✅ Document management features

**No prob-llama!** 🦙