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
  /** Number of chunks in vector database (0 = not processed, >0 = processed) */
  chunk_count?: number | null
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
  /** Whether uploads should auto-process by default (from config, optional) */
  auto_process?: boolean | null
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
  /** Type of action to execute */
  action_type: 'process' | 'delete_file_chunks' | 'delete_dataset_chunks'
  /** File hash for delete_file_chunks action */
  file_hash?: string
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
 * Response from deleting chunks for a file
 */
export interface DeleteChunksResponse {
  /** Status message */
  message: string
  /** File hash */
  file_hash: string
  /** Number of chunks deleted */
  deleted_chunks: number
}

/**
 * Response from deleting chunks for all files
 */
export interface DeleteAllChunksResponse {
  /** Status message */
  message: string
  /** Total number of chunks deleted */
  total_deleted_chunks: number
  /** Number of files whose chunks were deleted */
  total_files_cleared: number
  /** Number of files that failed to have chunks deleted */
  total_files_failed: number
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
  /** Current status of the upload/processing */
  status?: 'processing' | 'uploaded' | 'skipped'
  /** Processing task id when auto-processing is triggered */
  task_id?: string | null
}

/**
 * Response from bulk uploading files to a dataset
 */
export interface BulkFileUploadResponse {
  /** Number of files successfully uploaded */
  uploaded: number
  /** Number of files skipped as duplicates */
  skipped: number
  /** Number of files that failed to upload */
  failed: number
  /** Current status of the bulk operation */
  status: 'processing' | 'uploaded'
  /** Processing task id when auto-processing is triggered */
  task_id?: string | null
}

/**
 * Response from deleting a file from a dataset
 */
export interface FileDeleteResponse {
  /** Hash of the deleted file */
  file_hash: string
  /** Number of chunks deleted from vector store */
  deleted_chunks: number
}

/**
 * Response from checking task status
 */
export interface TaskStatusResponse {
  /** Task identifier */
  task_id: string
  /** Current task state (matches Celery task states) */
  state: 'PENDING' | 'STARTED' | 'SUCCESS' | 'FAILURE' | 'RETRY' | 'REVOKED'
  /** Task metadata (progress info, etc.) */
  meta: any | null
  /** Task result when successful */
  result: any | null
  /** Error message when failed */
  error: string | null
  /** Error traceback when failed */
  traceback: string | null
  /** Whether the task was cancelled */
  cancelled?: boolean
}

/**
 * Cleanup error details
 */
export interface CleanupError {
  /** File hash that failed to revert */
  file_hash: string
  /** Error message */
  error: string
}

/**
 * Response from cancelling a task
 */
export interface CancelTaskResponse {
  /** Human-readable message about the cancellation */
  message: string
  /** The ID of the cancelled task */
  task_id: string
  /** Whether the task was successfully cancelled */
  cancelled: boolean
  /** Number of pending tasks that were cancelled */
  pending_tasks_cancelled: number
  /** Number of running tasks at the time of cancellation */
  running_tasks_at_cancel: number
  /** Number of files that were successfully reverted */
  files_reverted: number
  /** Number of files that failed to revert */
  files_failed_to_revert: number
  /** List of errors encountered during cleanup */
  errors?: CleanupError[] | null
  /** True if the task had already completed before cancellation was requested */
  already_completed: boolean
  /** True if the task was already cancelled */
  already_cancelled: boolean
}

/**
 * Query parameters for file deletion
 */
export interface FileDeleteParams {
  /** Whether to remove the file from disk (optional) */
  remove_from_disk?: boolean
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
