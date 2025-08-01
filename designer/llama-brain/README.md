# 🧠 Llama Brain - Meta AI Configuration Assistant for LlamaFarm

Llama Brain is a **meta AI system** that uses LlamaFarm's own systems to help users configure LlamaFarm. It's a self-referential configuration assistant that uses its own hardcoded configurations when calling LlamaFarm CLIs to generate user configurations.

## 🔄 The Meta Architecture

**Key Insight**: Llama Brain uses its own YAML configurations (`/configs/llama_brain_*.yaml`) when invoking LlamaFarm CLI commands to help users create their own LlamaFarm configurations.

When you ask "How do I configure a RAG system?", Llama Brain:
1. Uses `llama_brain_rag.yaml` to search its knowledge base
2. Finds examples from ingested `default.yaml` files
3. Uses `llama_brain_models.yaml` to generate responses with llama3.2:3b
4. Uses `llama_brain_prompts.yaml` for specialized configuration prompts
5. Calls actual LlamaFarm CLIs with these configs to validate generated configs

## 🚀 Quick Start

### Prerequisites

1. **UV Package Manager** (required)
   ```bash
   # Install UV if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Ollama** (required)
   ```bash
   # Start Ollama server
   ollama serve
   ```

### One-Command Setup

```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain
./bootstrap.sh
```

This bootstrap script will:
- ✅ Verify Ollama is running
- 📥 Pull required models (llama3.2:3b, nomic-embed-text)
- 📦 Install dependencies with UV
- 🗂️ Initialize data directories
- 📚 Ingest documentation and default configs
- 🧪 Test the complete system

### Quick Commands After Setup

**Terminal 1 - Start Llama Brain Server:**
```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain
uv run --active python -m llama_brain.server.main
# Server runs on http://localhost:8080
```

**Terminal 2 - Start Flask Chat UI:**
```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain/test-chat-app
python app.py
# UI runs on http://localhost:5001
```

**Open Browser:** http://localhost:5001

## 📦 Manual Setup Instructions

### 1. Install Dependencies with UV

```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain

# Create virtual environment and install dependencies
uv sync

# Activate the environment (if needed for manual commands)
source .venv/bin/activate
```

### 2. Pull Required Models

```bash
# Chat model
ollama pull llama3.2:3b

# Embedding model for RAG
ollama pull nomic-embed-text
```

### 3. Run Setup Script

```bash
# Run setup with UV
uv run --active python setup.py
```

This will:
- Test LlamaFarm CLI availability
- Validate hardcoded configurations
- Ingest documentation and default.yaml files into RAG
- Test the complete system integration

### 4. Start the Server

```bash
# Start Llama Brain server with UV
uv run --active python -m llama_brain.server.main
```

The server will start on `http://localhost:8080`

### 5. Start the Test Chat UI

In a new terminal:

```bash
cd /Users/robthelen/llamafarm-1/designer/llama-brain/test-chat-app

# Install Flask if needed
pip install flask

# Start the chat app
python app.py
```

Open your browser to `http://localhost:5001`

**Note**: Port 5000 is often used by macOS AirPlay Receiver. The Flask app is configured to use port 5001 instead.

## 🎯 Usage Examples

### In the Chat Interface

Try these example queries:

**Model Configuration:**
- "Help me create a model config for local development"
- "I need a production setup with fallbacks"
- "Show me how to configure Ollama"

**RAG Configuration:**
- "Create a RAG system for PDF documents"
- "I need high-accuracy document search for legal files"
- "Set up a knowledge base for markdown docs"

**Prompt Configuration:**
- "Help me create medical domain prompts"
- "I need customer support prompt templates"
- "Show me A/B testing setup for prompts"

### Via API

```bash
# Chat endpoint (streamlined pipeline)
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a RAG config for technical documentation",
    "session_id": "test-session"
  }'

# System status
curl http://localhost:8080/status
```

## 📁 Project Structure

```
llama-brain/
├── configs/                        # Hardcoded YAML configurations
│   ├── llama_brain_models.yaml    # Ollama + llama3.2:3b config
│   ├── llama_brain_rag.yaml       # ChromaDB + embedding config
│   └── llama_brain_prompts.yaml   # Configuration assistant prompts
├── llama_brain/
│   ├── integrations/
│   │   └── llamafarm_client.py    # Core integration with LlamaFarm CLIs
│   ├── agents/                    # Specialized configuration agents
│   ├── chat/                      # Streamlined chat pipeline
│   └── server/                    # FastAPI REST API
├── test-chat-app/                 # Flask-based test UI
├── data/                          # Runtime data (created on setup)
│   ├── rag/                       # RAG database storage
│   ├── chat/                      # Chat session history
│   └── configs/                   # Generated user configs
├── setup.py                       # Setup script using UV
├── bootstrap.sh                   # One-command bootstrap
└── pyproject.toml                 # UV project configuration
```

## 🛠️ All UV Commands

### Setup Commands

```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Update dependencies
uv sync --upgrade

# Show installed packages
uv pip list
```

### Running Commands

```bash
# Run setup script
uv run --active python setup.py

# Start server
uv run --active python -m llama_brain.server.main

# Run any Python script
uv run --active python <script.py>

# Run with specific Python version
uv run --python 3.11 python <script.py>
```

### Development Commands

```bash
# Run tests (if you add them)
uv run pytest

# Run linting
uv run ruff check .

# Format code
uv run ruff format .

# Type checking
uv run mypy llama_brain/
```

## 🔍 Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Verify models are available
ollama list
```

### UV Issues

```bash
# Ensure UV is in PATH
which uv

# If not found, add to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Recreate virtual environment
rm -rf .venv
uv sync
```

### RAG Database Issues

```bash
# Clear and rebuild RAG database
rm -rf data/rag/chroma_db
uv run --active python setup.py
```

### Permission Issues

```bash
# Make scripts executable
chmod +x bootstrap.sh
chmod +x setup.py

# Fix directory permissions
chmod -R 755 data/
```

## 🧪 Testing the System

### 1. Verify Setup

```bash
# Check system status
curl http://localhost:8080/status
```

### 2. Test RAG Search

```bash
# Search documentation
cd ../../rag
uv run --active python cli.py --config ../designer/llama-brain/configs/llama_brain_rag.yaml \
  search "model configuration" --top-k 3
```

### 3. Test Model Validation

```bash
# Validate model config
cd ../../models
uv run --active python cli.py validate-config \
  ../designer/llama-brain/configs/llama_brain_models.yaml
```

### 4. Test Prompt Templates

```bash
# List available templates
cd ../../prompts
uv run --active python -m prompts.cli template list \
  --config ../designer/llama-brain/configs/llama_brain_prompts.yaml
```

## 📚 How It Works

### The Streamlined Pipeline

When you send a message to `/chat`:

1. **Intent Detection**: Analyzes your request
2. **Prompt Selection**: Chooses specialized configuration prompt
3. **RAG Context**: Searches knowledge base for relevant examples
4. **LLM Generation**: Uses llama3.2:3b with context
5. **Agent Triggering**: Automatically invokes appropriate agent
6. **Config Creation**: Generates and validates configuration
7. **Response**: Returns explanation + working config file

### Knowledge Base Contents

The RAG system ingests:
- All LlamaFarm README files
- `/rag/config/default.yaml` - Comprehensive RAG examples
- `/models/config/default.yaml` - All provider configurations
- `/prompts/config/default.yaml` - Template examples

### Meta Configuration Usage

Each LlamaFarm CLI call uses Llama Brain's own configs:
- Models CLI → uses `llama_brain_models.yaml`
- RAG CLI → uses `llama_brain_rag.yaml`
- Prompts CLI → uses `llama_brain_prompts.yaml`

## 🎨 Customization

### Modify Hardcoded Configs

Edit files in `/configs/` to:
- Change the chat model (default: llama3.2:3b)
- Use different embeddings (default: nomic-embed-text)
- Adjust RAG parameters
- Customize prompt templates

### Add Custom Knowledge

1. Add document paths to `llama_brain_rag.yaml`
2. Re-run setup: `uv run python setup.py`

### Extend Agents

Add new agents in `llama_brain/agents/` for:
- Additional LlamaFarm components
- Custom configuration logic
- External integrations

## 🤝 Contributing

1. Use UV for dependency management
2. Follow existing code patterns
3. Test with actual LlamaFarm CLIs
4. Update documentation

## 📄 License

[Include your license here]

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Review system logs
3. Verify LlamaFarm CLI availability
4. Ensure Ollama models are downloaded

---

**Remember**: Llama Brain is meta - it uses itself to configure itself to help you configure LlamaFarm! 🤯