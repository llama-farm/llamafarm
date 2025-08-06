# HuggingFace Embedder

**Framework:** HuggingFace Transformers

**When to use:** Use specific models from HuggingFace hub, local or API-based.

**Schema fields:**
- `model_name`: HF model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
- `use_api`: Use HF Inference API vs local
- `api_token`: HuggingFace API token (if using API)
- `device`: Computing device (cpu, cuda, mps)
- `max_length`: Maximum token length

**Best practices:**
- Choose model based on language/domain
- Use local for privacy
- Use API for convenience
- Match device to hardware