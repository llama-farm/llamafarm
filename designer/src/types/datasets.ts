/**
 * Dataset API Types - aligned with server/api/routers/datasets/
 *
 * This file contains types for Dataset API communication.
 * These types should remain stable and aligned with the API contract.
 */

/**
 * File information with metadata
 */
export interface DatasetFile {
  /** File content hash */
  hash: string
  /** Original filename */
  original_file_name: string
  /** Resolved filename */
  resolved_file_name: string
  /** File size in bytes */
  size: number
  /** MIME type of the file */
  mime_type: string
  /** Upload timestamp */
  timestamp: number
}

/**
 * Core Dataset entity structure for API communication
 * Used in responses from the datasets service
 */
export interface Dataset {
  /** Dataset name within the project */
  name: string
  /** Data processing strategy used for processing */
  data_processing_strategy: string
  /** Database used for processing */
  database: string
  /** Array of file hashes included in this dataset */
  files: string[]
  /** Extra details about the dataset */
  details?: DatasetDetails
}

/**
 * Extra details about the dataset
 */
export interface DatasetDetails {
  /** Array of file details */
  files_metadata: DatasetFile[]
}

/**
 * Request payload for creating a new dataset
 */
export interface CreateDatasetRequest {
  /** Dataset name */
  name: string
  /** RAG strategy to use for processing */
  data_processing_strategy: string
  /** Database to use for processing */
  database: string
}

/**
 * Response from creating a new dataset
 */
export interface CreateDatasetResponse {
  /** The created dataset */
  dataset: Dataset
}

/**
 * Response from listing datasets in a project
 */
export interface ListDatasetsResponse {
  /** Total number of datasets */
  total: number
  /** Array of datasets with flexible file format */
  datasets: Dataset[]
}

/**
 * Response from deleting a dataset
 */
export interface DeleteDatasetResponse {
  /** The deleted dataset */
  dataset: Dataset
}

/**
 * Request payload for executing dataset actions
 */
export interface DatasetActionRequest {
  /** Type of action to execute (currently only 'process') */
  action_type: 'process'
}

/**
 * Response from executing a dataset action
 */
export interface DatasetActionResponse {
  /** Status message */
  message: 'Accepted'
  /** URI for tracking the task */
  task_uri: string
  /** Task identifier */
  task_id: string
}

/**
 * Response from uploading a file to a dataset
 */
export interface FileUploadResponse {
  /** Name of the uploaded file */
  filename: string
  /** Hash of the uploaded file */
  hash: string
  /** Whether the file has been processed */
  processed: boolean
  /** Whether the file was skipped (duplicate) */
  skipped: boolean
}

/**
 * Response from deleting a file from a dataset
 */
export interface FileDeleteResponse {
  /** Hash of the deleted file */
  file_hash: string
}

/**
 * Response from checking task status
 */
export interface TaskStatusResponse {
  /** Task identifier */
  task_id: string
  /** Current task state */
  state: 'PENDING' | 'SUCCESS' | 'FAILURE' | 'RETRY' | 'REVOKED'
  /** Task metadata (progress info, etc.) */
  meta: any | null
  /** Task result when successful */
  result: any | null
  /** Error message when failed */
  error: string | null
  /** Error traceback when failed */
  traceback: string | null
}

/**
 * Parameters for API calls that require path parameters
 */
export interface DatasetPathParams {
  /** Project namespace */
  namespace: string
  /** Project identifier */
  project: string
  /** Dataset name */
  dataset?: string
  /** File hash for file-specific operations */
  file_hash?: string
}

/**
 * Query parameters for file deletion
 */
export interface FileDeleteParams {
  /** Whether to remove the file from disk (optional) */
  remove_from_disk?: boolean
}

/**
 * Standard error response structure
 */
export interface DatasetApiError {
  /** Error detail message */
  detail?: string
}

/**
 * Base error classes for Dataset API operations
 */
export class DatasetError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public data?: any
  ) {
    super(message)
    this.name = 'DatasetError'
  }
}

export class DatasetValidationError extends DatasetError {
  constructor(message: string, data?: any) {
    super(message, 422, data)
    this.name = 'DatasetValidationError'
  }
}

export class DatasetNotFoundError extends DatasetError {
  constructor(message: string, data?: any) {
    super(message, 404, data)
    this.name = 'DatasetNotFoundError'
  }
}

export class DatasetNetworkError extends DatasetError {
  constructor(message: string, originalError?: any) {
    super(message, undefined, originalError)
    this.name = 'DatasetNetworkError'
  }
}

/**
 * Raw async task detail format from server: [success: bool, info: object]
 */
export type RawFileProcessingDetail = [
  boolean, // success
  {
    filename?: string
    file_hash?: string
    status?: string
    reason?: string
    parser?: string
    extractors?: string[]
    chunks?: number | null
    chunk_size?: number | null
    embedder?: string
    error?: string
    stored_count?: number
    skipped_count?: number
    result?: {
      status?: string
      filename?: string
      reason?: string
      document_count?: number
      stored_count?: number
      skipped_count?: number
      parsers_used?: string[]
      embedder?: string
      extractors_applied?: string[]
      document_ids?: string[]
    }
  },
]

/**
 * Normalized file processing detail (converted from RawFileProcessingDetail)
 */
export interface FileProcessingDetail {
  /** File hash identifier */
  hash: string
  /** Original filename (may differ from hash if hash is a SHA) */
  filename: string
  /** Whether processing was successful */
  success: boolean
  /** Processing status: processed, skipped, or failed */
  status: string
  /** Parser used for this file */
  parser?: string
  /** Extractors applied to this file */
  extractors?: string[]
  /** Number of chunks created */
  chunks?: number
  /** Chunk size used */
  chunk_size?: number
  /** Embedder used */
  embedder?: string
  /** Error message if failed */
  error?: string
  /** Reason for skipped status */
  reason?: string
  /** Number of chunks stored in vector database */
  stored_count?: number
  /** Number of chunks skipped (duplicates) */
  skipped_count?: number
}

/**
 * Response from processing a dataset
 */
export interface ProcessDatasetResponse {
  /** Status message */
  message: string
  /** Number of files successfully processed */
  processed_files: number
  /** Number of files skipped (e.g., duplicates) */
  skipped_files: number
  /** Number of files that failed processing */
  failed_files: number
  /** Data processing strategy used */
  strategy?: string | null
  /** Database used */
  database?: string | null
  /** Detailed results for each file */
  details: FileProcessingDetail[]
  /** Task ID for async processing */
  task_id?: string | null
}

/**
 * Frontend-specific Dataset type with additional UI properties
 * This extends the API Dataset type with local state management fields
 */
export interface UIDataset extends Dataset {
  /** Local UI identifier */
  id: string
  /** Last run timestamp for UI display */
  lastRun: Date
  /** Embedding model used */
  embedModel: string
  /** Number of chunks processed */
  numChunks: number
  /** Processing percentage (0-100) */
  processedPercent: number
  /** Version identifier */
  version: string
  /** Optional description */
  description?: string
}

/**
 * Frontend File representation for UI components
 */
export interface UIFile {
  /** Stable identifier for the file */
  id: string
  /** File name */
  name: string
  /** File size in bytes */
  size: number
  /** Last modified timestamp */
  lastModified: number
  /** MIME type (optional) */
  type?: string
}
