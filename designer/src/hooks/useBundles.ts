import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listBundles,
  deleteBundle,
  estimateBundleSize,
  type BundleRequest,
} from '../api/bundleService'

export const bundleKeys = {
  all: ['bundles'] as const,
  list: () => [...bundleKeys.all, 'list'] as const,
}

export function useBundles() {
  return useQuery({
    queryKey: bundleKeys.list(),
    queryFn: listBundles,
    staleTime: 30_000,
  })
}

export function useDeleteBundle() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteBundle(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: bundleKeys.list() })
    },
  })
}

export function useEstimateBundleSize() {
  return useMutation({
    mutationFn: (req: BundleRequest) => estimateBundleSize(req),
  })
}
