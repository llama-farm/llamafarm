# Basic Similarity Retriever

**Framework:** Base retriever

**When to use:** Simple similarity search without additional processing.

**Schema fields:**
- `top_k`: Number of results to return
- `score_threshold`: Minimum similarity score
- `include_metadata`: Return document metadata

**Best practices:**
- Start with top_k=10
- Set threshold based on testing
- Always include metadata
- Good baseline retriever