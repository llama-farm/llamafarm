import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import datasetService from '../api/datasets'
import {
  CreateDatasetRequest,
  DatasetActionRequest,
  FileDeleteParams,
} from '../types/datasets'
import { projectKeys } from './useProjects'

export const DEFAULT_DATASET_LIST_STALE_TIME = 600_000 // 10 minutes

/**
 * Query keys for dataset-related queries
 * Follows the pattern used in existing project hooks
 */
export const datasetKeys = {
  all: ['datasets'] as const,
  lists: () => [...datasetKeys.all, 'list'] as const,
  list: (namespace: string, project: string) =>
    [...datasetKeys.lists(), namespace, project] as const,
  details: () => [...datasetKeys.all, 'detail'] as const,
  detail: (namespace: string, project: string, dataset: string) =>
    [...datasetKeys.details(), namespace, project, dataset] as const,
  strategies: (namespace: string, project: string) =>
    [...datasetKeys.all, 'strategies', namespace, project] as const,
}

/**
 * Hook to fetch available strategies and databases for a project
 * @param namespace - The namespace containing the project
 * @param project - The project to fetch strategies for
 * @param options - Additional query options
 * @returns Query result with available strategies and databases
 */
export function useAvailableStrategies(
  namespace: string,
  project: string,
  options?: {
    enabled?: boolean
    staleTime?: number
  }
) {
  return useQuery({
    queryKey: datasetKeys.strategies(namespace, project),
    queryFn: () => datasetService.getAvailableStrategies(namespace, project),
    enabled: options?.enabled !== false && !!namespace && !!project,
    staleTime: options?.staleTime ?? DEFAULT_DATASET_LIST_STALE_TIME,
  })
}

/**
 * Hook to fetch all datasets in a project
 * @param namespace - The namespace containing the project
 * @param project - The project to fetch datasets for
 * @param options - Additional query options
 * @returns Query result with datasets list
 */
export function useListDatasets(
  namespace: string,
  project: string,
  options?: {
    enabled?: boolean
    staleTime?: number
  }
) {
  return useQuery({
    queryKey: datasetKeys.list(namespace, project),
    queryFn: () => datasetService.listDatasets(namespace, project),
    enabled: options?.enabled !== false && !!namespace && !!project,
    staleTime: options?.staleTime ?? DEFAULT_DATASET_LIST_STALE_TIME,
  })
}

/**
 * Hook to create a new dataset
 * @returns Mutation for creating datasets
 */
export function useCreateDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (
      data: { namespace: string; project: string } & CreateDatasetRequest
    ) =>
      datasetService.createDataset(data.namespace, data.project, {
        name: data.name,
        data_processing_strategy: data.data_processing_strategy,
        database: data.database,
      }),
    onSuccess: (_, variables) => {
      // Invalidate and refetch the datasets list
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
      // Also invalidate the project config so ConfigEditor shows the new dataset
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to delete a dataset
 * @returns Mutation for deleting datasets
 */
export function useDeleteDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
    }) =>
      datasetService.deleteDataset(data.namespace, data.project, data.dataset),
    onSuccess: (_, variables) => {
      // Invalidate and refetch the datasets list
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
      // Also invalidate the project config so ConfigEditor shows the new dataset
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to execute dataset actions (like ingest)
 * @returns Mutation for executing dataset actions
 */
export function useDatasetAction() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
      request: DatasetActionRequest
    }) =>
      datasetService.executeDatasetAction(
        data.namespace,
        data.project,
        data.dataset,
        data.request
      ),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh any status changes
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to ingest a dataset (convenience wrapper for dataset action)
 * @returns Mutation for ingesting datasets
 */
export function useIngestDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
    }) =>
      datasetService.ingestDataset(data.namespace, data.project, data.dataset),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh any status changes
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to upload files to a dataset
 * @returns Mutation for uploading files
 */
export function useUploadFileToDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
      file: File
      signal?: AbortSignal
    }) =>
      datasetService.uploadFileToDataset(
        data.namespace,
        data.project,
        data.dataset,
        data.file,
        data.signal
      ),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh file counts
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to delete files from a dataset
 * @returns Mutation for deleting files
 */
export function useDeleteFileFromDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
      fileHash: string
      params?: FileDeleteParams
    }) =>
      datasetService.deleteFileFromDataset(
        data.namespace,
        data.project,
        data.dataset,
        data.fileHash,
        data.params
      ),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh file counts
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to upload multiple files to a dataset sequentially
 * @returns Mutation for uploading multiple files
 */
export function useUploadMultipleFiles() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (data: {
      namespace: string
      project: string
      dataset: string
      files: File[]
    }) => {
      const results = []
      for (const file of data.files) {
        const result = await datasetService.uploadFileToDataset(
          data.namespace,
          data.project,
          data.dataset,
          file
        )
        results.push(result)
      }
      return results
    },
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh file counts
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to check task status with automatic polling
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param taskId - The task identifier (null to disable)
 * @param options - Additional query options
 * @returns Query result with task status
 */
export function useTaskStatus(
  namespace: string,
  project: string,
  taskId: string | null,
  options?: {
    enabled?: boolean
    refetchInterval?: number
  }
) {
  return useQuery({
    queryKey: ['task-status', namespace, project, taskId],
    queryFn: () => datasetService.getTaskStatus(namespace, project, taskId!),
    enabled: !!taskId && !!namespace && !!project && options?.enabled !== false,
    refetchInterval: (query) => {
      // Stop polling if task completed or failed
      const data = query.state.data as any
      if (data?.state === 'SUCCESS' || data?.state === 'FAILURE') {
        return false
      }
      return options?.refetchInterval || 2000 // Poll every 2 seconds by default
    },
    refetchIntervalInBackground: true, // Continue polling even when tab is not focused
    staleTime: 0, // Always consider stale to ensure fresh polling
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes after unmount
  })
}

/**
 * Hook to re-ingest a dataset (trigger reprocessing)
 * @returns Mutation for re-ingesting datasets
 */
export function useReIngestDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
    }) =>
      datasetService.executeDatasetAction(
        data.namespace,
        data.project,
        data.dataset,
        { action_type: 'ingest' }
      ),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh any status changes
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to process a dataset (async processing with task tracking)
 * @returns Mutation for processing datasets
 */
export function useProcessDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
    }) =>
      datasetService.processDataset(
        data.namespace,
        data.project,
        data.dataset
      ),
    onSuccess: (_, variables) => {
      // Invalidate datasets list to refresh any status changes
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })
    },
  })
}

/**
 * Hook to delete a file from a dataset
 * @returns Mutation for deleting files from datasets
 */
export function useDeleteDatasetFile() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: {
      namespace: string
      project: string
      dataset: string
      fileHash: string
      removeFromDisk?: boolean
    }) =>
      datasetService.deleteDatasetFile(
        data.namespace,
        data.project,
        data.dataset,
        data.fileHash,
        data.removeFromDisk ?? true
      ),
    onSuccess: (result, variables) => {
      // Invalidate datasets list to refresh the files list
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
      })

      // Result contains the deleted file details which can be accessed
      // by the component for more specific user feedback
      if (result?.file_hash) {
        // File deletion confirmed - the component can access this via the mutation result
      }
    },
    onError: (error, variables) => {
      console.error(`Failed to delete file ${variables.fileHash}:`, error)
    },
  })
}

/**
 * Default export with all dataset hooks
 */
export default {
  useAvailableStrategies,
  useListDatasets,
  useCreateDataset,
  useDeleteDataset,
  useDatasetAction,
  useIngestDataset,
  useUploadFileToDataset,
  useDeleteFileFromDataset,
  useUploadMultipleFiles,
  useTaskStatus,
  useReIngestDataset,
  useDeleteDatasetFile,
  datasetKeys,
}
