---
title: Advanced Retrieval Strategies
sidebar_position: 7
---

# Advanced Retrieval Strategies

LlamaFarm supports advanced retrieval strategies that improve result quality through reranking and query decomposition.

## Cross-Encoder Reranking

### Overview

Cross-encoder reranking improves retrieval accuracy by reranking initial candidates using a specialized model. This is **10-100x faster** than LLM-based reranking while often providing better relevance scores.

**How it works:**
1. Initial retrieval gets 30+ candidates using fast vector search
2. Cross-encoder model jointly encodes query+document pairs
3. Results are reranked by relevance scores
4. Top N results returned

**Performance:**
- Speed: Fast (50-400 docs/sec)
- Accuracy: Very High
- Best for: Simple, focused questions requiring accurate ranking

### Configuration

Add a reranker model to your runtime configuration:

```yaml
runtime:
  models:
    - name: reranker
      description: Cross-encoder model for semantic reranking
      provider: openai
      model: qllama/bce-reranker-base_v1:q4_k_m
      base_url: http://localhost:11434/v1
```

Then configure the retrieval strategy in your database:

```yaml
rag:
  databases:
    - name: main_database
      type: ChromaStore
      retrieval_strategies:
        - name: reranked_search
          type: CrossEncoderRerankedStrategy
          config:
            model_name: reranker  # References runtime.models
            initial_k: 30         # Initial candidates
            final_k: 5            # Final results after reranking
            base_strategy: BasicSimilarityStrategy
            base_strategy_config:
              distance_metric: cosine
            batch_size: 16
            normalize_scores: true
            relevance_threshold: 0.0
            max_chars_per_doc: 1000
          default: true
```

### Recommended Models

Pull via Ollama:

```bash
# Best for production (multilingual, high accuracy)
ollama pull qllama/bge-reranker-v2-m3:q4_k_m

# Good balance (smaller, faster)
ollama pull qllama/bce-reranker-base_v1:q4_k_m
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `reranker` | Name of model from `runtime.models` |
| `initial_k` | 30 | Number of candidates before reranking |
| `final_k` | 10 | Number of results after reranking |
| `base_strategy` | `BasicSimilarityStrategy` | Initial retrieval strategy |
| `batch_size` | 32 | Reranking batch size |
| `normalize_scores` | `true` | Normalize scores to 0-1 range |
| `relevance_threshold` | 0.0 | Minimum score to include |
| `max_chars_per_doc` | 1000 | Truncate long documents |

### Usage

```bash
# Query with reranking (if set as default)
lf rag query --database main_database "What are the differences between llama and alpaca fibers?"

# Explicitly specify strategy
lf rag query --database main_database --retrieval-strategy reranked_search "your question"
```

## Multi-Turn RAG

### Overview

Multi-turn RAG handles complex, multi-part queries by:
1. Detecting query complexity
2. Decomposing complex queries into focused sub-queries
3. Retrieving documents for each sub-query in parallel
4. Optionally reranking each sub-query's results
5. Merging and deduplicating final results

**Performance:**
- Speed: Medium-slow (1-3 seconds)
- Accuracy: Very High for complex queries
- Best for: Context-heavy questions with multiple aspects

### When to Use

**Good candidates for multi-turn RAG:**
- "What are the differences between X and Y, and how does this affect Z?"
- "Compare A and B, and also explain their applications"
- "What is X? How does it work? What are the benefits?"

**Not ideal for:**
- Short, simple questions ("What is AI?")
- Single-aspect queries ("Describe neural networks")

### Configuration

Add a query decomposition model:

```yaml
runtime:
  models:
    - name: query_decomposer
      description: Model for query decomposition
      provider: openai
      model: gemma3:1b  # Small, fast model works well
      base_url: http://localhost:11434/v1
```

Configure the multi-turn strategy:

```yaml
rag:
  databases:
    - name: main_database
      retrieval_strategies:
        - name: multi_turn_search
          type: MultiTurnRAGStrategy
          config:
            model_name: query_decomposer
            max_sub_queries: 3
            complexity_threshold: 50  # Min chars to trigger decomposition
            min_query_length: 20
            base_strategy: BasicSimilarityStrategy
            base_strategy_config:
              distance_metric: cosine
            sub_query_top_k: 10       # Results per sub-query
            final_top_k: 5            # Final merged results
            # Enable reranking for each sub-query
            enable_reranking: true
            reranker_strategy: CrossEncoderRerankedStrategy
            reranker_config:
              model_name: reranker
              initial_k: 15
              final_k: 10
              batch_size: 16
            dedup_similarity_threshold: 0.95
            max_workers: 3            # Parallel sub-query retrieval
          default: true
```

### Complexity Detection

Queries are considered complex if they:
- Are longer than `complexity_threshold` characters (default: 50)
- Contain patterns like: `and`, `also`, `additionally`, `furthermore`, `moreover`
- Have multiple question marks

**Examples:**

✅ Complex (will decompose):
- "What are the current paradigm limitations in AI development, and how do centralized cloud-based models compare to local AI deployment?" (178 chars + "and")
- "Explain attention mechanisms and also describe transformers" (62 chars + "also")

❌ Simple (won't decompose):
- "What is AI?" (12 chars)
- "Explain machine learning" (25 chars, no patterns)

### Recommended Models

**Query Decomposition:**
```bash
# Good balance: speed vs. format adherence
ollama pull gemma3:1b

# Alternative (faster, smaller)
ollama pull qwen3:1.7B
```

**Reranking (optional but recommended):**
```bash
ollama pull qllama/bce-reranker-base_v1:q4_k_m
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | - | Name of decomposition model from `runtime.models` |
| `max_sub_queries` | 3 | Maximum sub-queries to generate |
| `complexity_threshold` | 50 | Min chars to trigger decomposition |
| `min_query_length` | 20 | Min length for each sub-query |
| `sub_query_top_k` | 10 | Results per sub-query |
| `final_top_k` | 10 | Final results after merging |
| `enable_reranking` | `false` | Enable reranking per sub-query |
| `max_workers` | 3 | Parallel workers for sub-queries |
| `dedup_similarity_threshold` | 0.95 | Deduplication threshold |

### Usage

```bash
# Complex query (will decompose)
lf rag query --database main_database \
  "What are the current paradigm limitations in AI development, and how do centralized cloud-based models compare to local AI deployment in terms of cost and performance?"

# The system will:
# 1. Detect complexity (length + "and" pattern)
# 2. Decompose into sub-queries like:
#    - "What are current paradigm limitations in AI development?"
#    - "How do centralized cloud-based AI models work?"
#    - "What are the costs and performance of local AI deployment?"
# 3. Retrieve for each sub-query in parallel
# 4. Rerank each sub-query's results (if enabled)
# 5. Merge and deduplicate to top 5 results
```

### Monitoring

Check RAG server logs to see decomposition in action:

```
[info] Query complexity detected (reason='pattern: \band\b')
[info] Query is complex, decomposing into sub-queries
[info] Decomposed query into 3 sub-queries
[info] Retrieving for 3 sub-queries in parallel
[info] Merging and deduplicating results
```

## Combining Strategies

You can configure multiple strategies in a single database:

```yaml
rag:
  databases:
    - name: main_database
      retrieval_strategies:
        # Fast, basic search
        - name: basic_search
          type: BasicSimilarityStrategy
          config:
            top_k: 10
          default: false

        # Accurate reranked search for simple queries
        - name: reranked_search
          type: CrossEncoderRerankedStrategy
          config:
            model_name: reranker
            initial_k: 30
            final_k: 5
          default: false

        # Comprehensive search for complex queries
        - name: multi_turn_search
          type: MultiTurnRAGStrategy
          config:
            model_name: query_decomposer
            enable_reranking: true
            reranker_config:
              model_name: reranker
          default: true  # Use this by default
```

Then specify the strategy at query time:

```bash
# Use default (multi-turn)
lf rag query --database main_database "complex multi-part question"

# Use basic search for speed
lf rag query --database main_database --retrieval-strategy basic_search "simple question"

# Use reranking for accuracy
lf rag query --database main_database --retrieval-strategy reranked_search "focused question"
```

## Performance Tips

### Cross-Encoder Reranking

1. **Adjust `initial_k` and `final_k`:**
   - Higher `initial_k` = better recall, slower
   - Lower `final_k` = faster LLM processing
   - Good defaults: `initial_k: 30`, `final_k: 5`

2. **Batch size:**
   - Larger batches = faster (up to a point)
   - Default `32` works well for most models

3. **Document truncation:**
   - `max_chars_per_doc: 1000` prevents token limit errors
   - Adjust based on your documents and model

### Multi-Turn RAG

1. **Complexity threshold:**
   - Lower (e.g., 30) = more queries decomposed, slower but thorough
   - Higher (e.g., 100) = only very complex queries decomposed, faster
   - Default `50` is a good balance

2. **Parallel workers:**
   - More workers = faster parallel retrieval
   - Limited by CPU cores and model concurrency
   - Default `3` works well

3. **Reranking toggle:**
   - `enable_reranking: true` = highest accuracy, slower
   - `enable_reranking: false` = faster, still good quality
   - Use reranking for critical applications

## Troubleshooting

### Cross-Encoder Issues

**Problem:** Error loading reranker model
```
ValueError: Model 'reranker' not found in runtime.models
```

**Solution:** Ensure the model exists in `runtime.models` and the name matches:
```yaml
runtime:
  models:
    - name: reranker  # Must match model_name in strategy config
      model: qllama/bce-reranker-base_v1:q4_k_m
```

**Problem:** Reranking is slow

**Solution:**
- Reduce `initial_k` (fewer candidates to rerank)
- Increase `batch_size` (more efficient batching)
- Use a smaller model (e.g., bce-reranker vs bge-reranker-v2-m3)

### Multi-Turn Issues

**Problem:** Query decomposition returns original query (no decomposition)
```
[warning] No <question> tags found in LLM response
```

**Solution:**
- Use a more capable model (gemma3:1b recommended over qwen3-0.6B)
- Check model is running: `ollama list`
- Lower temperature if model is too creative

**Problem:** Sub-queries are nonsensical

**Solution:**
- Increase temperature slightly (0.3 recommended)
- Try a different model
- Check your complexity threshold isn't too low

**Problem:** Too slow for production

**Solution:**
- Increase `complexity_threshold` to decompose fewer queries
- Disable `enable_reranking` or use it selectively
- Reduce `max_sub_queries` to 2
- Reduce `sub_query_top_k` to 5

## Next Steps

- [RAG Guide](./index.md) - Core RAG concepts
- [CLI Reference](../cli/index.md) - Command usage
- [Configuration Guide](../configuration/index.md) - Full YAML reference
- [Examples](../examples/index.md) - See real-world configurations
