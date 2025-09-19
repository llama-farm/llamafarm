// Shared retrieval strategy helpers

export type StrategyType =
  | 'BasicSimilarityStrategy'
  | 'MetadataFilteredStrategy'
  | 'MultiQueryStrategy'
  | 'RerankedStrategy'
  | 'HybridUniversalStrategy'

export type HybridSub = {
  id: string
  type: Exclude<StrategyType, 'HybridUniversalStrategy'>
  weight: string
  config?: Record<string, unknown>
}

export function getDefaultConfigForRetrieval(
  type: Exclude<StrategyType, 'HybridUniversalStrategy'>
): Record<string, unknown> {
  switch (type) {
    case 'BasicSimilarityStrategy':
      return { top_k: 10, distance_metric: 'cosine', score_threshold: null }
    case 'MetadataFilteredStrategy':
      return {
        top_k: 10,
        filters: {},
        filter_mode: 'pre',
        fallback_multiplier: 3,
      }
    case 'MultiQueryStrategy':
      return {
        num_queries: 3,
        top_k: 10,
        aggregation_method: 'weighted',
        query_weights: null,
      }
    case 'RerankedStrategy':
      return {
        initial_k: 30,
        final_k: 10,
        rerank_factors: {
          similarity_weight: 0.7,
          recency_weight: 0.1,
          length_weight: 0.1,
          metadata_weight: 0.1,
        },
        normalize_scores: true,
      }
  }
}

export function parseWeightsList(
  raw: string,
  expectedLength?: number
): number[] | null {
  const parts = raw
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
  if (parts.length === 0) return null
  const weights = parts.map(p => Number(p))
  if (weights.some(w => !Number.isFinite(w) || w < 0 || w > 1)) return null
  if (typeof expectedLength === 'number' && expectedLength > 0) {
    if (weights.length !== expectedLength) return null
  }
  return weights
}
