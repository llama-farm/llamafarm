---
title: Retrieval Strategies Reference
sidebar_position: 6
---

# Retrieval Strategies Reference

Retrieval strategies control how documents are searched and ranked. LlamaFarm provides strategies from basic similarity search to advanced multi-stage retrieval.

## Quick Start

Retrieval strategies are configured in `retrieval_strategies` within a database:

```yaml
rag:
  databases:
    - name: main_db
      retrieval_strategies:
        - name: semantic_search
          type: BasicSimilarityStrategy
          config:
            top_k: 10
          default: true
```

### Common Properties

| Property | Required | Description |
|----------|----------|-------------|
| `name` | Yes | Strategy identifier |
| `type` | Yes | Strategy type |
| `config` | No | Strategy-specific options |
| `default` | No | Whether this is the default strategy |

---

## BasicSimilarityStrategy

Simple vector similarity search. Fast and effective for most use cases.

**Best for:** General semantic search, simple queries

```yaml
- name: semantic_search
  type: BasicSimilarityStrategy
  config:
    top_k: 10
    distance_metric: cosine
    score_threshold: 0.5
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `top_k` | integer | 10 | Number of results (1-1000) |
| `distance_metric` | string | `cosine` | `cosine`, `euclidean`, `manhattan`, `dot` |
| `score_threshold` | number | `null` | Minimum similarity score (0-1) |

---

## MetadataFilteredStrategy

Filter results using document metadata before or after retrieval.

**Best for:** Faceted search, filtering by date/type/category

```yaml
- name: filtered_search
  type: MetadataFilteredStrategy
  config:
    top_k: 10
    filter_mode: pre
    filters:
      doc_type: report
      year: 2024
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `top_k` | integer | 10 | Number of results (1-1000) |
| `filters` | object | `{}` | Metadata key-value filters |
| `filter_mode` | string | `pre` | `pre` (before retrieval) or `post` (after) |
| `fallback_multiplier` | integer | 3 | Multiplier for post-filtering |

### Filter Examples

```yaml
filters:
  # Exact match
  doc_type: report

  # Multiple values (OR)
  category: [finance, legal]

  # Numeric
  year: 2024
```

---

## MultiQueryStrategy

Generate multiple query variations for better recall.

**Best for:** Complex queries, improving recall

```yaml
- name: multi_query
  type: MultiQueryStrategy
  config:
    num_queries: 3
    top_k: 10
    aggregation_method: weighted
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_queries` | integer | 3 | Query variations (1-10) |
| `top_k` | integer | 10 | Results per query (1-1000) |
| `aggregation_method` | string | `weighted` | `max`, `mean`, `weighted`, `reciprocal_rank` |
| `query_weights` | array | `null` | Weights for each query |

### Aggregation Methods

| Method | Description |
|--------|-------------|
| `max` | Take highest score per document |
| `mean` | Average scores across queries |
| `weighted` | Weight by query importance |
| `reciprocal_rank` | RRF fusion of rankings |

---

## RerankedStrategy

Two-stage retrieval with score-based reranking.

**Best for:** Balancing recall and precision

```yaml
- name: reranked
  type: RerankedStrategy
  config:
    initial_k: 30
    final_k: 10
    normalize_scores: true
    rerank_factors:
      similarity_weight: 0.7
      recency_weight: 0.2
      length_weight: 0.1
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `initial_k` | integer | 30 | Initial candidates (10-1000) |
| `final_k` | integer | 10 | Final results (1-100) |
| `normalize_scores` | boolean | `true` | Normalize before combining |
| `rerank_factors` | object | See below | Weighting factors |

### Rerank Factors

```yaml
rerank_factors:
  similarity_weight: 0.7   # Vector similarity
  recency_weight: 0.1      # Document freshness
  length_weight: 0.1       # Content length
  metadata_weight: 0.1     # Metadata relevance
```

---

## HybridUniversalStrategy

Combine multiple retrieval strategies with score fusion.

**Best for:** Complex retrieval needs, combining sparse and dense

```yaml
- name: hybrid
  type: HybridUniversalStrategy
  config:
    final_k: 10
    combination_method: weighted_average
    strategies:
      - type: BasicSimilarityStrategy
        weight: 0.6
        config:
          top_k: 20
      - type: MetadataFilteredStrategy
        weight: 0.4
        config:
          top_k: 20
          filters:
            doc_type: report
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strategies` | array | - | Sub-strategies to combine (2-5) |
| `combination_method` | string | `weighted_average` | Fusion method |
| `final_k` | integer | 10 | Final results (1-1000) |

### Combination Methods

| Method | Description |
|--------|-------------|
| `weighted_average` | Weighted average of scores |
| `rank_fusion` | Reciprocal rank fusion |
| `score_fusion` | Direct score fusion |

### Sub-Strategy Format

```yaml
strategies:
  - type: BasicSimilarityStrategy  # Strategy type
    weight: 0.6                     # Relative weight (0-1)
    config:                         # Strategy config
      top_k: 20
```

**Available sub-strategy types:**
- `BasicSimilarityStrategy`
- `MetadataFilteredStrategy`
- `MultiQueryStrategy`
- `RerankedStrategy`

---

## CrossEncoderRerankedStrategy

Neural reranking using cross-encoder models for highest accuracy.

**Best for:** High-precision search, quality-critical applications

```yaml
- name: reranked_search
  type: CrossEncoderRerankedStrategy
  config:
    model_name: reranker
    initial_k: 30
    final_k: 5
    relevance_threshold: 0.3
```

**Requires a reranker model in runtime:**

```yaml
runtime:
  models:
    - name: reranker
      provider: universal
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      base_url: http://127.0.0.1:11540
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model_name` | string | - | Yes | Name of model from `runtime.models` |
| `initial_k` | integer | 30 | No | Initial candidates (10-100) |
| `final_k` | integer | 10 | No | Final results (1-50) |
| `base_strategy` | string | `BasicSimilarityStrategy` | No | Initial retrieval strategy |
| `base_strategy_config` | object | `{}` | No | Config for base strategy |
| `relevance_threshold` | number | 0.0 | No | Minimum relevance (0-1) |
| `timeout` | integer | 60 | No | Request timeout in seconds |

### Recommended Models

| Model | Size | Speed | Multilingual |
|-------|------|-------|--------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 90MB | Very Fast | No |
| `BAAI/bge-reranker-base` | 280MB | Fast | Yes |
| `BAAI/bge-reranker-v2-m3` | 560MB | Medium | Yes |

See [Advanced Retrieval Strategies](./advanced-retrieval.md) for detailed model recommendations.

---

## MultiTurnRAGStrategy

Query decomposition for complex, multi-part questions.

**Best for:** Complex queries, multi-aspect questions

```yaml
- name: multi_turn
  type: MultiTurnRAGStrategy
  config:
    model_name: query_decomposer
    max_sub_queries: 3
    complexity_threshold: 50
    final_top_k: 10
    enable_reranking: true
```

**Requires a decomposition model in runtime:**

```yaml
runtime:
  models:
    - name: query_decomposer
      provider: openai
      model: gemma3:1b
      base_url: http://localhost:11434/v1
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model_name` | string | - | Yes | Decomposition model name |
| `max_sub_queries` | integer | 3 | No | Max sub-queries (1-5) |
| `complexity_threshold` | integer | 50 | No | Min chars to trigger decomposition |
| `min_query_length` | integer | 20 | No | Min length per sub-query |
| `base_strategy` | string | `BasicSimilarityStrategy` | No | Strategy for sub-queries |
| `base_strategy_config` | object | `{}` | No | Base strategy config |
| `sub_query_top_k` | integer | 10 | No | Results per sub-query |
| `final_top_k` | integer | 10 | No | Final merged results |
| `enable_reranking` | boolean | `false` | No | Rerank each sub-query |
| `reranker_strategy` | string | `CrossEncoderRerankedStrategy` | No | Reranking strategy |
| `reranker_config` | object | `{}` | No | Reranker configuration |
| `dedup_similarity_threshold` | number | 0.95 | No | Dedup threshold (0.5-1) |
| `max_workers` | integer | 3 | No | Parallel workers (1-10) |

See [Advanced Retrieval Strategies](./advanced-retrieval.md) for detailed configuration.

---

## Additional Retriever Types

These types are also available for use in database configurations:

| Type | Description |
|------|-------------|
| `VectorRetriever` | Basic vector search |
| `HybridRetriever` | Hybrid sparse/dense |
| `BM25Retriever` | Keyword-based BM25 |
| `RerankedRetriever` | Generic reranker |
| `GraphRetriever` | Graph-based retrieval |
| `ElasticRetriever` | Elasticsearch backend |

---

## Strategy Selection Guide

| Query Type | Recommended Strategy | Why |
|------------|---------------------|-----|
| Simple, specific | `BasicSimilarityStrategy` | Fast, effective |
| Filtered search | `MetadataFilteredStrategy` | Precise filtering |
| Broad topics | `MultiQueryStrategy` | Better recall |
| Quality-critical | `CrossEncoderRerankedStrategy` | Highest accuracy |
| Complex questions | `MultiTurnRAGStrategy` | Query decomposition |
| Mixed requirements | `HybridUniversalStrategy` | Combine approaches |

---

## Multiple Strategies Example

Configure multiple strategies and switch at query time:

```yaml
retrieval_strategies:
  # Fast default
  - name: fast
    type: BasicSimilarityStrategy
    config:
      top_k: 10
    default: true

  # Filtered by type
  - name: by_type
    type: MetadataFilteredStrategy
    config:
      top_k: 10
      filter_mode: pre

  # High accuracy
  - name: accurate
    type: CrossEncoderRerankedStrategy
    config:
      model_name: reranker
      initial_k: 30
      final_k: 5

  # Complex queries
  - name: complex
    type: MultiTurnRAGStrategy
    config:
      model_name: query_decomposer
      enable_reranking: true

default_retrieval_strategy: fast
```

Query with specific strategies:

```bash
# Use default (fast)
lf rag query --database main_db "simple question"

# Use accurate strategy
lf rag query --database main_db --retrieval-strategy accurate "important question"

# Use complex for multi-part questions
lf rag query --database main_db --retrieval-strategy complex \
  "What are the benefits of X and how does it compare to Y?"
```

## Next Steps

- [Advanced Retrieval Strategies](./advanced-retrieval.md) - Detailed cross-encoder and multi-turn setup
- [Databases Reference](./databases.md) - Configure vector stores
- [RAG Guide](./index.md) - Full RAG overview
