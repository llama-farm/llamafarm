import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import datasetService from '../api/datasets'
import {
  CreateDatasetRequest,
  DatasetActionRequest,
  FileDeleteParams,
} from '../types/datasets'

/**
 * Query keys for dataset-related queries
 * Follows the pattern used in existing project hooks
 */
export const datasetKeys = {
  all: ['datasets'] as const,
  lists: () => [...datasetKeys.all, 'list'] as const,
  list: (namespace: string, project: string) => [...datasetKeys.lists(), namespace, project] as const,
  details: () => [...datasetKeys.all, 'detail'] as const,
  detail: (namespace: string, project: string, dataset: string) => [...datasetKeys.details(), namespace, project, dataset] as const,
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
    staleTime: options?.staleTime ?? 60_000, // 1 minute default
  })
}

/**
 * Hook to create a new dataset
 * @returns Mutation for creating datasets
 */
export function useCreateDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: { namespace: string; project: string } & CreateDatasetRequest) =>
      datasetService.createDataset(data.namespace, data.project, {
        name: data.name,
        rag_strategy: data.rag_strategy,
      }),
    onSuccess: (_, variables) => {
      // Invalidate and refetch the datasets list
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
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
    mutationFn: (data: { namespace: string; project: string; dataset: string }) =>
      datasetService.deleteDataset(data.namespace, data.project, data.dataset),
    onSuccess: (_, variables) => {
      // Invalidate and refetch the datasets list
      queryClient.invalidateQueries({
        queryKey: datasetKeys.list(variables.namespace, variables.project),
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
    mutationFn: (data: { namespace: string; project: string; dataset: string }) =>
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
    }) =>
      datasetService.uploadFileToDataset(
        data.namespace,
        data.project,
        data.dataset,
        data.file
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
 * Default export with all dataset hooks
 */
export default {
  useListDatasets,
  useCreateDataset,
  useDeleteDataset,
  useDatasetAction,
  useIngestDataset,
  useUploadFileToDataset,
  useDeleteFileFromDataset,
  useUploadMultipleFiles,
  datasetKeys,
}
