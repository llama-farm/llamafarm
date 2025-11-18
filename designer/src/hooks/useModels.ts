import { useQuery } from '@tanstack/react-query'
import modelService from '../api/modelService'

/**
 * Query keys for model-related queries
 */
export const modelKeys = {
  all: ['models'] as const,
  cached: (provider?: string) => [...modelKeys.all, 'cached', provider] as const,
}

/**
 * Hook to fetch cached models from disk
 * @param provider - The provider to list models for (default: universal)
 * @returns Query result with cached models list
 */
export const useCachedModels = (provider = 'universal') => {
  return useQuery({
    queryKey: modelKeys.cached(provider),
    queryFn: () => modelService.listCachedModels(provider),
    staleTime: 5 * 60 * 1000, // Consider data fresh for 5 minutes
    retry: 1, // Only retry once on failure
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
  })
}

