# Multi Query Retriever

**Framework:** Query expansion

**When to use:** Improve recall by generating multiple query variations.

**Schema fields:**
- `num_queries`: Number of query variations
- `query_generator`: Method to generate queries
- `aggregate_method`: How to combine results
- `diversity_ranker`: Diversify final results

**Best practices:**
- Generate 3-5 queries
- Use LLM for query generation
- Remove duplicate results
- Balance diversity and relevance