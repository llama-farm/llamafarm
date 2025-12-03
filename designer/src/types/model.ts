/**
 * Model types for LlamaFarm Designer
 */

export interface Model {
  name: string // Internal name (e.g., "fast", "powerful")
  model: string // Actual model ID (e.g., "gemma3:1b")
  provider: string // Provider (e.g., "ollama", "lemonade")
  description?: string // Optional description
  default: boolean // Whether this is the default model
  base_url?: string // Base URL for the provider
  prompt_format?: string // Prompt format
  // Lemonade-specific fields
  lemonade?: {
    backend?: string
    port?: number
    context_size?: number
  }
}

export interface ListModelsResponse {
  total: number
  models: Model[]
}

/**
 * Cached model information from the backend (disk models)
 */
export interface CachedModel {
  id: string
  name: string
  size: number
  path: string
}

/**
 * Response from the list cached models endpoint
 */
export interface ListCachedModelsResponse {
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
  | { event: 'warning'; message: string }

/**
 * Response from the delete model endpoint
 */
export interface DeleteModelResponse {
  model_name: string
  revisions_deleted: number
  size_freed: number
  path: string
}

/**
 * Response from the validate download endpoint
 */
export interface ValidateDownloadResponse {
  can_download: boolean
  warning: boolean
  message: string
  available_bytes: number
  required_bytes: number
  cache_info: {
    total_bytes: number
    used_bytes: number
    free_bytes: number
    path: string
    percent_free: number
  }
  system_info: {
    total_bytes: number
    used_bytes: number
    free_bytes: number
    path: string
    percent_free: number
  }
}
