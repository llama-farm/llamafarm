# Qdrant Store

**Framework:** Qdrant

**When to use:** High-performance vector search with rich filtering capabilities.

**Schema fields:**
- `url`: Qdrant server URL
- `api_key`: API key (if using cloud)
- `collection_name`: Collection name
- `vector_size`: Vector dimension
- `distance`: Distance metric
- `on_disk`: Store vectors on disk

**Best practices:**
- Use local for development
- Use cloud for production
- Enable on_disk for large datasets
- Leverage payload filtering