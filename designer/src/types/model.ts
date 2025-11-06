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
 * These events are emitted by the server during model download via SSE
 */
export type DownloadEvent =
  | { event: 'start'; desc: string; total: number | null; n: number }
  | { event: 'progress'; desc: string; total: number | null; n: number }
  | { event: 'end'; desc: string; total: number | null; n: number }
  | { event: 'done'; local_dir: string }
  | { event: 'error'; message: string }

/**
 * Aggregated download progress for UI display
 */
export interface DownloadProgress {
  currentFile: string
  currentFileProgress: number // 0-100
  currentFileDownloaded: number // bytes
  currentFileTotal: number // bytes
  filesCompleted: number
  totalFiles: number
  overallDownloaded: number // bytes
  overallTotal: number // bytes
  overallProgress: number // 0-100
  downloadSpeed: number // bytes/second
  estimatedTimeRemaining: number // seconds
  elapsedTime: number // seconds
}

/**
 * Response from the delete model endpoint
 */
export interface DeleteModelResponse {
  model_name: string
  revisions_deleted: number
  size_freed: number
  path: string
}
