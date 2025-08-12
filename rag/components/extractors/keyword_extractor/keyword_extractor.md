# Keyword Extractor

**Framework:** YAKE, RAKE, or TextRank

**When to use:** Extract important keywords and key phrases for indexing, tagging, or summarization.

**Schema fields:**
- `algorithm`: Extraction algorithm (yake, rake, textrank)
- `max_keywords`: Maximum keywords to extract
- `min_score`: Minimum relevance score
- `deduplication_threshold`: Similarity threshold for deduplication
- `include_scores`: Return keyword scores

**Best practices:**
- YAKE for statistical extraction
- RAKE for rapid automatic extraction
- TextRank for graph-based extraction
- Adjust max_keywords based on document length
- Use scores for ranking importance