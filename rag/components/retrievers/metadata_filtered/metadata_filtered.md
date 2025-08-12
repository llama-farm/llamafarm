# Metadata Filtered Retriever

**Framework:** Metadata-based filtering

**When to use:** Filter results by metadata before similarity search.

**Schema fields:**
- `filters`: Metadata filter conditions
- `top_k`: Number of results after filtering
- `pre_filter`: Apply filters before search
- `operator`: Combine filters (AND, OR)

**Best practices:**
- Pre-filter for performance
- Use specific metadata fields
- Combine with similarity search
- Index filtered fields