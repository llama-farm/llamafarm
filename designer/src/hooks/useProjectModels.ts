import { useQuery } from '@tanstack/react-query'
import modelService from '../api/modelService'

export const modelKeys = {
  all: ['models'] as const,
  lists: () => [...modelKeys.all, 'list'] as const,
  list: (namespace: string, projectId: string) =>
    [...modelKeys.lists(), namespace, projectId] as const,
}

export const useProjectModels = (
  namespace: string | undefined,
  projectId: string | undefined,
  enabled = true
) => {
  return useQuery({
    queryKey: modelKeys.list(namespace || '', projectId || ''),
    queryFn: () => modelService.listModels(namespace!, projectId!),
    enabled: enabled && !!namespace && !!projectId,
    staleTime: 60_000,
  })
}

export default useProjectModels
