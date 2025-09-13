# LlamaFarm Examples

Clean, practical examples demonstrating LlamaFarm capabilities.

## Available Examples

### 📚 RAG Pipeline (`rag_pipeline/`)
A complete example of the Retrieval-Augmented Generation pipeline showing:
- Document ingestion and processing
- Embedding generation with Ollama
- Vector storage with ChromaDB
- Semantic search and retrieval

**Quick Start:**
```bash
cd rag_pipeline
python rag_example.py
```

## Prerequisites

All examples require:
1. LlamaFarm installed (`pip install llamafarm`)
2. Ollama running (`ollama serve`)
3. Required models pulled (see example READMEs)

## Structure

Each example includes:
- **Sample data** - Real-world documents for testing
- **Configuration** - Working `llamafarm.yaml` config
- **Python script** - Runnable example code
- **README** - Detailed documentation

## Contributing

Have a great example? We welcome contributions! Please ensure your example:
- Demonstrates a specific use case clearly
- Includes sample data and configuration
- Has a focused README with prerequisites
- Runs without additional setup (beyond standard requirements)

## Support

- [Documentation](https://docs.llamafarm.com)
- [GitHub Issues](https://github.com/llamafarm/llamafarm/issues)
- [Discord Community](https://discord.gg/llamafarm)