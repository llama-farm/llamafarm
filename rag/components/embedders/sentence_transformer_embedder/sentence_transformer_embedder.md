# Sentence Transformer Embedder

**Framework:** sentence-transformers

**When to use:** Fast, efficient sentence embeddings with many model options.

**Schema fields:**
- `model_name`: Model name (e.g., "all-MiniLM-L6-v2")
- `device`: Computing device (cpu, cuda)
- `normalize_embeddings`: L2 normalize vectors
- `batch_size`: Encoding batch size
- `show_progress`: Show progress bar

**Best practices:**
- Use all-MiniLM-L6-v2 for English
- Use multilingual models for other languages
- Normalize for cosine similarity
- GPU for faster processing