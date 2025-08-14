---
title: Retrieval strategies
sidebar_label: Retrieval strategies
slug: /rag/retrieval-strategies
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Database-agnostic strategies that automatically optimize for your store.

## BasicSimilarityStrategy (getting started)

```json
{
  "retrieval_strategy": {
    "type": "BasicSimilarityStrategy",
    "config": { "distance_metric": "cosine" }
  }
}
```

Use cases: simple semantic search; Performance: fast; Complexity: low.

## MetadataFilteredStrategy (smart filtering)

```json
{
  "retrieval_strategy": {
    "type": "MetadataFilteredStrategy",
    "config": {
      "distance_metric": "cosine",
      "default_filters": {
        "priority": ["high", "medium"],
        "type": "documentation"
      },
      "fallback_multiplier": 3
    }
  }
}
```

Features: native filtering when supported, automatic fallback, complex operators. Use cases: domain searches, multi-tenant. Performance: medium.

## MultiQueryStrategy (enhanced recall)

```json
{
  "retrieval_strategy": {
    "type": "MultiQueryStrategy",
    "config": {
      "num_queries": 3,
      "aggregation_method": "weighted",
      "search_multiplier": 2
    }
  }
}
```

Use cases: ambiguous queries, query expansion. Performance: medium.

## RerankedStrategy (sophisticated ranking)

```json
{
  "retrieval_strategy": {
    "type": "RerankedStrategy",
    "config": {
      "initial_k": 20,
      "length_normalization": 1000,
      "rerank_factors": {
        "recency": 0.1,
        "length": 0.05,
        "metadata_boost": 0.2
      }
    }
  }
}
```

Use cases: production systems; Performance: slower; Complexity: high.

## HybridUniversalStrategy (best of all worlds)

```json
{
  "retrieval_strategy": {
    "type": "HybridUniversalStrategy",
    "config": {
      "combination_method": "weighted_average",
      "normalize_scores": true,
      "diversity_boost": 0.1,
      "strategies": [
        { "type": "BasicSimilarityStrategy", "weight": 0.4 },
        { "type": "MetadataFilteredStrategy", "weight": 0.3 },
        { "type": "RerankedStrategy", "weight": 0.2 },
        { "type": "MultiQueryStrategy", "weight": 0.1 }
      ]
    }
  }
}
```

## Strategy selection guide

| Use Case           | Recommended              | Why                                   |
| ------------------ | ------------------------ | ------------------------------------- |
| Getting Started    | BasicSimilarityStrategy  | Simple, fast baseline                 |
| Production General | HybridUniversalStrategy  | Balanced performance                  |
| High Precision     | RerankedStrategy         | Multi-factor ranking                  |
| Filtered Content   | MetadataFilteredStrategy | Efficient domain-specific searches    |
| Complex Queries    | MultiQueryStrategy       | Better recall for ambiguous questions |
