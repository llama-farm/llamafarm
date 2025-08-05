# Ollama Embedder

**Framework:** Ollama (local)

**When to use:** Local embedding generation with privacy and no API costs.

**Schema fields:**
- `model`: Model name (e.g., "nomic-embed-text")
- `api_base`: Ollama API URL (default: http://localhost:11434)
- `batch_size`: Documents per batch
- `timeout`: Request timeout in seconds

**Best practices:**
- Use nomic-embed-text for quality
- Adjust batch size for memory
- Run Ollama locally for privacy
- Monitor model availability