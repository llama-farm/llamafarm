import { apiClient } from './client'
import {
  ListDatasetsResponse,
  CreateDatasetRequest,
  CreateDatasetResponse,
  DeleteDatasetResponse,
  DatasetActionRequest,
  DatasetActionResponse,
  FileUploadResponse,
  FileDeleteResponse,
  FileDeleteParams,
  TaskStatusResponse,
} from '../types/datasets'

/**
 * Get available strategies and databases for a project
 * @param namespace - The namespace to get strategies for
 * @param project - The project to get strategies for
 * @returns Promise with available strategies and databases
 */
export async function getAvailableStrategies(
  namespace: string,
  project: string
): Promise<{ data_processing_strategies: string[]; databases: string[] }> {
  const response = await apiClient.get(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/strategies`
  )
  return response.data
}

/**
 * List all datasets in a project
 * @param namespace - The namespace to list datasets for
 * @param project - The project to list datasets for
 * @returns Promise<ListDatasetsResponse> - List of datasets with total count
 */
export async function listDatasets(
  namespace: string,
  project: string
): Promise<ListDatasetsResponse> {
  const response = await apiClient.get<ListDatasetsResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/`
  )
  return response.data
}

/**
 * Create a new dataset in a project
 * @param namespace - The namespace containing the project
 * @param project - The project to create the dataset in
 * @param request - The dataset creation request
 * @returns Promise<CreateDatasetResponse> - The created dataset
 */
export async function createDataset(
  namespace: string,
  project: string,
  request: CreateDatasetRequest
): Promise<CreateDatasetResponse> {
  // The server expects { name, data_processing_strategy, database }
  // Use the provided request values directly (no rag_strategy translation).
  let data_processing_strategy: string = request.data_processing_strategy
  let database: string = request.database

  try {
    const strategiesResp = await apiClient.get(
      `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/strategies`
    )
    const strategies: string[] =
      strategiesResp.data?.data_processing_strategies || []
    const databases: string[] = strategiesResp.data?.databases || []

    if (strategies.length > 0) {
      if (!strategies.includes(data_processing_strategy)) {
        throw new Error(
          `Invalid data_processing_strategy: ${data_processing_strategy}. Allowed: ${strategies.join(', ')}`
        )
      }
    }
    if (databases.length > 0) {
      if (!databases.includes(database)) {
        throw new Error(
          `Invalid database: ${database}. Allowed: ${databases.join(', ')}`
        )
      }
    }
  } catch (err) {
    // If the strategies endpoint is unavailable, proceed with provided defaults.
    // But if it was available and we threw due to invalid strategy, rethrow.
    if (
      err instanceof Error &&
      err.message.startsWith('Invalid data_processing_strategy')
    ) {
      throw err
    }
  }

  const serverPayload = {
    name: request.name,
    data_processing_strategy,
    database,
  }

  const response = await apiClient.post<CreateDatasetResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/`,
    serverPayload
  )
  return response.data
}

/**
 * Delete a dataset from a project
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name to delete
 * @returns Promise<DeleteDatasetResponse> - The deleted dataset
 */
export async function deleteDataset(
  namespace: string,
  project: string,
  dataset: string
): Promise<DeleteDatasetResponse> {
  const response = await apiClient.delete<DeleteDatasetResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}`
  )
  return response.data
}

/**
 * Execute an action on a dataset (currently only 'ingest')
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @param request - The action request
 * @returns Promise<DatasetActionResponse> - The action response with task URI
 */
export async function executeDatasetAction(
  namespace: string,
  project: string,
  dataset: string,
  request: DatasetActionRequest
): Promise<DatasetActionResponse> {
  const response = await apiClient.post<DatasetActionResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/actions`,
    request
  )
  return response.data
}

/**
 * Upload a file to a dataset using multipart form data
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @param file - The file to upload
 * @param signal - Optional AbortSignal for cancelling the upload
 * @returns Promise<FileUploadResponse> - The upload response with filename
 */
export async function uploadFileToDataset(
  namespace: string,
  project: string,
  dataset: string,
  file: File,
  signal?: AbortSignal
): Promise<FileUploadResponse> {
  const formData = new FormData()

  // Strip directory paths from filename before uploading
  // This handles folder uploads where file.name might be "folder/file.pdf"
  const cleanFileName = file.name.split('/').pop() || file.name

  // Create new File with clean name if needed
  const fileToUpload = file.name !== cleanFileName
    ? new File([file], cleanFileName, { type: file.type, lastModified: file.lastModified })
    : file

  formData.append('file', fileToUpload)

  const response = await apiClient.post<FileUploadResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/data`,
    formData,
    {
      signal,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  )
  return response.data
}

/**
 * Delete a file from a dataset
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @param fileHash - The hash of the file to delete
 * @param params - Optional query parameters
 * @returns Promise<FileDeleteResponse> - The delete response with file hash
 */
export async function deleteFileFromDataset(
  namespace: string,
  project: string,
  dataset: string,
  fileHash: string,
  params?: FileDeleteParams
): Promise<FileDeleteResponse> {
  const queryParams = new URLSearchParams()
  if (params?.remove_from_disk !== undefined) {
    queryParams.append('remove_from_disk', params.remove_from_disk.toString())
  }

  const url = `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/data/${encodeURIComponent(fileHash)}`
  const fullUrl = queryParams.toString()
    ? `${url}?${queryParams.toString()}`
    : url

  const response = await apiClient.delete<FileDeleteResponse>(fullUrl)
  return response.data
}

/**
 * Convenience function to ingest a dataset (execute 'ingest' action)
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @returns Promise<DatasetActionResponse> - The action response with task URI
 */
export async function ingestDataset(
  namespace: string,
  project: string,
  dataset: string
): Promise<DatasetActionResponse> {
  return executeDatasetAction(namespace, project, dataset, {
    action_type: 'ingest',
  })
}

/**
 * Get the status of a task
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param taskId - The task identifier
 * @returns Promise<TaskStatusResponse> - The task status information
 */
export async function getTaskStatus(
  namespace: string,
  project: string,
  taskId: string
): Promise<TaskStatusResponse> {
  const response = await apiClient.get<TaskStatusResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/tasks/${encodeURIComponent(taskId)}`
  )
  return response.data
}

/**
 * Delete a file from a dataset
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @param fileHash - The file hash to delete
 * @param removeFromDisk - Whether to also remove the file from disk
 * @returns Promise<FileDeleteResponse> - The delete response with file hash
 */
export async function deleteDatasetFile(
  namespace: string,
  project: string,
  dataset: string,
  fileHash: string,
  removeFromDisk: boolean = true
): Promise<FileDeleteResponse> {
  const queryParams = new URLSearchParams()
  if (removeFromDisk !== undefined) {
    queryParams.append('remove_from_disk', removeFromDisk.toString())
  }

  const url = `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/data/${encodeURIComponent(fileHash)}`
  const fullUrl = queryParams.toString()
    ? `${url}?${queryParams.toString()}`
    : url

  const response = await apiClient.delete<FileDeleteResponse>(fullUrl)
  return response.data
}

/**
 * Process a dataset (async processing with task tracking)
 * @param namespace - The project namespace
 * @param project - The project identifier
 * @param dataset - The dataset name
 * @returns Promise with task information
 */
export async function processDataset(
  namespace: string,
  project: string,
  dataset: string
): Promise<{ message: string; task_id: string }> {
  const response = await apiClient.post(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/process?async_processing=true`
  )
  return response.data
}

/**
 * Default export with all dataset service functions
 */
export default {
  getAvailableStrategies,
  listDatasets,
  createDataset,
  deleteDataset,
  executeDatasetAction,
  uploadFileToDataset,
  deleteFileFromDataset,
  ingestDataset,
  getTaskStatus,
  deleteDatasetFile,
  processDataset,
}
