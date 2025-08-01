# 🧠 Llama Brain Setup Guide

## Quick Setup (Recommended)

```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain
./bootstrap.sh
```

This will:
1. ✅ Check that Ollama is running
2. 📥 Pull required models (`llama3.2:3b`, `nomic-embed-text`)
3. 🗂️ Create data directories
4. ⚙️ Test hardcoded configurations
5. 📚 Ingest documentation + default configs into RAG database
6. 🧪 Test the complete system

## Manual Setup

If you prefer to run steps manually:

### 1. Prerequisites

**Start Ollama:**
```bash
ollama serve
```

**Pull required models:**
```bash
ollama pull llama3.2:3b        # Chat model
ollama pull nomic-embed-text   # Embedding model
```

### 2. Run Setup Script

```bash
python setup.py
```

### 3. Start the System

**Start Llama Brain server:**
```bash
python -m llama_brain.server.main
```

**Start test chat app (in another terminal):**
```bash
cd test-chat-app
python app.py
```

**Open browser:**
```
http://localhost:5000
```

## What Gets Set Up

### 🗂️ Data Directories
- `data/rag/` - RAG database and ChromaDB storage  
- `data/chat/` - Chat session history
- `data/configs/` - Generated user configurations

### 📚 Knowledge Base
The system ingests these files into the RAG database:
- **Documentation**: All README.md files from LlamaFarm components
- **Default Configs**: The comprehensive example configurations:
  - `/rag/config/default.yaml` - All RAG options and examples
  - `/models/config/default.yaml` - All model provider examples  
  - `/prompts/config/default.yaml` - All prompt template examples

### ⚙️ Hardcoded Configs
Llama Brain uses these configs when calling LlamaFarm CLIs:
- `configs/llama_brain_models.yaml` - Ollama + llama3.2:3b
- `configs/llama_brain_rag.yaml` - ChromaDB + nomic-embed-text
- `configs/llama_brain_prompts.yaml` - Specialized config assistant prompts

## Testing the System

Try these example queries in the chat interface:

### 🤖 Model Configuration
- "Help me create a model config for development"
- "I need a production setup with multiple providers"
- "Show me how to configure Ollama for local development"

### 🔍 RAG Configuration  
- "Create a RAG system for PDF documents"
- "I need to process legal documents with high accuracy"
- "How do I set up document search for my knowledge base?"

### 📝 Prompt Configuration
- "Help me create prompt templates for customer support"
- "I need domain-specific prompts for medical queries"
- "Show me how to set up A/B testing for prompts"

## The Meta Magic ✨

When you ask these questions, here's what happens:

1. **Your Question** → Llama Brain chat system
2. **RAG Search** → Uses `llama_brain_rag.yaml` to search knowledge base
3. **Knowledge Base** → Contains ingested default configs with examples
4. **LLM Response** → Uses `llama_brain_models.yaml` for llama3.2:3b  
5. **Agent Action** → Uses `llama_brain_prompts.yaml` for specialized responses
6. **Config Generation** → Calls actual LlamaFarm CLIs with hardcoded configs

**Result**: Llama Brain uses its own configurations to understand and generate LlamaFarm configurations! 🤯

## Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama if needed
ollama serve
```

### Missing Models
```bash
# List available models
ollama list

# Pull missing models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### RAG Database Issues
```bash
# Delete and recreate RAG database
rm -rf data/rag/chroma_db
python setup.py  # Re-run setup
```

### Permission Issues
```bash
# Make sure scripts are executable
chmod +x bootstrap.sh

# Check directory permissions
ls -la data/
```

## Advanced Usage

### Custom Knowledge Base
To add your own documentation to the knowledge base:

1. Add files to the `document_sources` list in `configs/llama_brain_rag.yaml`
2. Re-run ingestion: `python setup.py`

### Configuration Customization
The hardcoded configs in `/configs/` can be modified for:
- Different models (change from llama3.2:3b)
- Different vector stores (change from ChromaDB)
- Custom prompt templates

### Development Mode
For development, you can run individual components:

```bash
# Test RAG system only
cd ../../rag
python cli.py search "model configuration" --config ../designer/llama-brain/configs/llama_brain_rag.yaml

# Test models system only  
cd ../../models
python cli.py validate-config ../designer/llama-brain/configs/llama_brain_models.yaml

# Test prompts system only
python -m prompts.cli validate templates --config ../designer/llama-brain/configs/llama_brain_prompts.yaml
```