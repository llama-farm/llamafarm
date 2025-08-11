# FAISS Store

**Framework:** Facebook AI Similarity Search

**When to use:** High-performance similarity search for large-scale vectors.

**Schema fields:**
- `index_type`: Index type (Flat, IVF, HNSW)
- `dimension`: Vector dimension
- `metric`: Distance metric (L2, IP)
- `nlist`: Number of clusters (for IVF)
- `nprobe`: Clusters to search (for IVF)

**Best practices:**
- Use Flat for <10K vectors
- Use IVF for 10K-1M vectors
- Use HNSW for >1M vectors
- GPU acceleration when available