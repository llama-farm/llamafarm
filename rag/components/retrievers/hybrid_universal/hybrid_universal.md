# Hybrid Universal Retriever

**Framework:** Custom hybrid

**When to use:** Combine multiple retrieval strategies for better results.

**Schema fields:**
- `strategies`: List of retrieval strategies to combine
- `weights`: Weight for each strategy
- `fusion_method`: How to combine results (rrf, weighted)
- `top_k`: Final number of results

**Best practices:**
- Combine 2-3 strategies max
- Use RRF for balanced fusion
- Weight based on strategy strengths
- Test different combinations