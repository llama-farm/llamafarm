// Shared retrieval strategy catalog (labels/descriptions kept in sync across components)

export const STRATEGY_TYPES = [
  'BasicSimilarityStrategy',
  'MetadataFilteredStrategy',
  'MultiQueryStrategy',
  'RerankedStrategy',
  'HybridUniversalStrategy',
] as const

export type StrategyType = (typeof STRATEGY_TYPES)[number]

export const STRATEGY_LABELS: Record<StrategyType, string> = {
  BasicSimilarityStrategy: 'Basic similarity',
  MetadataFilteredStrategy: 'Metadata-filtered',
  MultiQueryStrategy: 'Multi-query',
  RerankedStrategy: 'Reranked',
  HybridUniversalStrategy: 'Hybrid universal',
}

export const STRATEGY_SLUG: Record<StrategyType, string> = {
  BasicSimilarityStrategy: 'basic-search',
  MetadataFilteredStrategy: 'metadata-filtered',
  MultiQueryStrategy: 'multi-query',
  RerankedStrategy: 'reranked',
  HybridUniversalStrategy: 'hybrid-universal',
}

export const STRATEGY_DESCRIPTIONS: Record<StrategyType, string> = {
  BasicSimilarityStrategy:
    'Simple, fast vector search. Returns the top matches by similarity (you set how many and the distance metric). Optionally filter out weak hits with a score threshold.',
  MetadataFilteredStrategy:
    'Search with filters like source, type, date, or tags. Choose whether filters apply before or after retrieval, and automatically widen results when post-filtering removes too much.',
  MultiQueryStrategy:
    'Ask the question several ways at once. We create multiple query variations and merge their results so you catch relevant content even when phrased differently.',
  RerankedStrategy:
    'Pull a larger candidate set first, then sort by quality. Tune weights for similarity, recency, length, and metadata; optionally normalize scores for fair comparisons.',
  HybridUniversalStrategy:
    'Blend multiple strategies into one result set. Combine with weighted average, rank fusion, or score fusion, then keep the best K.',
}
