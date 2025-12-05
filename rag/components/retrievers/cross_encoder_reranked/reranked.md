# Reranked Retriever

**Framework:** Two-stage retrieval

**When to use:** Improve precision by reranking initial results.

**Schema fields:**
- `initial_top_k`: Number of candidates to retrieve
- `final_top_k`: Number after reranking
- `reranker_model`: Model for reranking
- `cross_encoder`: Use cross-encoder model

**Best practices:**
- Retrieve 3-5x final count
- Use cross-encoder for accuracy
- Consider latency tradeoff
- Cache reranker model