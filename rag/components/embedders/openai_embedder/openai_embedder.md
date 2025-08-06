# OpenAI Embedder

**Framework:** OpenAI API

**When to use:** High-quality embeddings with OpenAI's models, when API costs are acceptable.

**Schema fields:**
- `model`: Model name (e.g., "text-embedding-3-small")
- `api_key`: OpenAI API key
- `dimensions`: Embedding dimensions (if model supports)
- `batch_size`: Documents per API call
- `max_retries`: Retry attempts on failure

**Best practices:**
- Use text-embedding-3-small for cost/quality balance
- Batch requests to reduce API calls
- Store API key securely
- Monitor usage for costs