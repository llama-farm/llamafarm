# Chroma Store

**Framework:** ChromaDB

**When to use:** Local vector storage with metadata filtering and persistence.

**Schema fields:**
- `collection_name`: Name of collection
- `distance_function`: Similarity metric (l2, ip, cosine)
- `batch_size`: Documents per insert batch

**Best practices:**
- Choose distance function based on embeddings
- Use metadata for filtering
- Batch inserts for performance