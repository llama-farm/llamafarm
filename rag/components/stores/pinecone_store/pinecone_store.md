# Pinecone Store

**Framework:** Pinecone (cloud)

**When to use:** Managed vector database for production workloads.

**Schema fields:**
- `api_key`: Pinecone API key
- `environment`: Pinecone environment
- `index_name`: Name of index
- `dimension`: Vector dimension
- `metric`: Similarity metric
- `namespace`: Optional namespace

**Best practices:**
- Use namespaces for multi-tenancy
- Monitor usage for costs
- Set appropriate pod type
- Use metadata for filtering