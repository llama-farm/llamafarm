/**
 * Cached model information from the backend
 */
export interface CachedModel {
  id: string
  name: string
  size: number
  path: string
}

/**
 * Response from the list models endpoint
 */
export interface ListModelsResponse {
  data: CachedModel[]
}

/**
 * Request to download a model
 */
export interface DownloadModelRequest {
  provider?: string
  model_name: string
}

/**
 * Events from the model download stream
 */
export type DownloadEvent =
  | { event: 'progress'; file?: string; downloaded: number; total: number }
  | { event: 'done'; local_dir: string }
  | { event: 'error'; message: string }

/**
 * Response from the delete model endpoint
 */
export interface DeleteModelResponse {
  model_name: string
  revisions_deleted: number
  size_freed: number
  path: string
}
