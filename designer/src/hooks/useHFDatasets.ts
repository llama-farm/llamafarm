/**
 * React Query hooks for Hugging Face datasets
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as hfApi from '../api/huggingface'
import type { HFDatasetImportRequest } from '../types/huggingface'
import { datasetKeys } from './useDatasets'
import { projectKeys } from './useProjects'

/**
 * Query keys for HF dataset-related queries
 */
export const hfDatasetKeys = {
  all: ['hf-datasets'] as const,
  search: (query: string) => [...hfDatasetKeys.all, 'search', query] as const,
}

/**
 * Hook to search HF datasets
 * @param query - Search query string
 * @param enabled - Whether the query is enabled (defaults to true when query has 2+ chars)
 * @returns Query result with search results
 */
export function useHFDatasetSearch(query: string, enabled = true) {
  return useQuery({
    queryKey: hfDatasetKeys.search(query),
    queryFn: () => hfApi.searchDatasets(query),
    enabled: enabled && query.length >= 2,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

/**
 * Hook to import HF dataset via backend
 * @returns Mutation for importing HF datasets
 */
export function useImportHFDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: HFDatasetImportRequest) => hfApi.importDataset(request),
    onSuccess: (data, variables) => {
      console.log('[useImportHFDataset] Import successful:', data)
      // Invalidate datasets list for the project
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
      // Also invalidate the project config so ConfigEditor and Data page show the new dataset
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(variables.namespace, variables.project),
      })
    },
    onError: (error, variables) => {
      console.error('[useImportHFDataset] Import failed:', error, variables)
    },
  })
}

export default {
  useHFDatasetSearch,
  useImportHFDataset,
  hfDatasetKeys,
}
