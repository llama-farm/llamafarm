/**
 * ML Service - API client for classifier and anomaly detection endpoints
 * Endpoints proxy to Universal Runtime via LlamaFarm server
 */

import { apiClient } from './client'
import type {
  // Classifier types
  ClassifierFitRequest,
  ClassifierFitResponse,
  ClassifierPredictRequest,
  ClassifierPredictResponse,
  ClassifierSaveRequest,
  ClassifierSaveResponse,
  ClassifierLoadRequest,
  ClassifierLoadResponse,
  ClassifierListModelsResponse,
  // Anomaly types
  AnomalyFitRequest,
  AnomalyFitResponse,
  AnomalyScoreRequest,
  AnomalyScoreResponse,
  AnomalySaveRequest,
  AnomalySaveResponse,
  AnomalyLoadRequest,
  AnomalyLoadResponse,
  AnomalyListModelsResponse,
  // Streaming Anomaly types
  StreamingAnomalyRequest,
  StreamingAnomalyResponse,
  StreamingDetectorStats,
  StreamingDetectorListResponse,
  // Document Scanning types
  DocumentScanningResponse,
  // Encoder types
  EmbeddingRequest,
  EmbeddingResponse,
  RerankRequest,
  RerankResponse,
  // Shared types
  MLHealthResponse,
  MLDeleteResponse,
} from '../types/ml'

// =============================================================================
// Health Check
// =============================================================================

/**
 * Check ML service health (Universal Runtime availability)
 */
export async function checkMLHealth(): Promise<MLHealthResponse> {
  const response = await apiClient.get<MLHealthResponse>('/ml/health')
  return response.data
}

// =============================================================================
// Classifier Endpoints
// =============================================================================

/**
 * Train a text classifier using SetFit few-shot learning
 */
export async function fitClassifier(
  request: ClassifierFitRequest
): Promise<ClassifierFitResponse> {
  const response = await apiClient.post<ClassifierFitResponse>(
    '/ml/classifier/fit',
    request
  )
  return response.data
}

/**
 * Classify texts using a trained classifier
 */
export async function predictClassifier(
  request: ClassifierPredictRequest
): Promise<ClassifierPredictResponse> {
  const response = await apiClient.post<ClassifierPredictResponse>(
    '/ml/classifier/predict',
    request
  )
  return response.data
}

/**
 * Save a trained classifier to disk
 */
export async function saveClassifier(
  request: ClassifierSaveRequest
): Promise<ClassifierSaveResponse> {
  const response = await apiClient.post<ClassifierSaveResponse>(
    '/ml/classifier/save',
    request
  )
  return response.data
}

/**
 * Load a pre-trained classifier from disk
 */
export async function loadClassifier(
  request: ClassifierLoadRequest
): Promise<ClassifierLoadResponse> {
  const response = await apiClient.post<ClassifierLoadResponse>(
    '/ml/classifier/load',
    request
  )
  return response.data
}

/**
 * List all saved classifier models
 */
export async function listClassifierModels(): Promise<ClassifierListModelsResponse> {
  const response = await apiClient.get<ClassifierListModelsResponse>('/ml/classifier/models')
  return response.data
}

/**
 * Delete a saved classifier model
 */
export async function deleteClassifierModel(
  modelName: string
): Promise<MLDeleteResponse> {
  const response = await apiClient.delete<MLDeleteResponse>(
    `/ml/classifier/models/${encodeURIComponent(modelName)}`
  )
  return response.data
}

// =============================================================================
// Anomaly Detection Endpoints
// =============================================================================

/**
 * Train an anomaly detection model
 */
export async function fitAnomaly(
  request: AnomalyFitRequest
): Promise<AnomalyFitResponse> {
  const response = await apiClient.post<AnomalyFitResponse>(
    '/ml/anomaly/fit',
    request
  )
  return response.data
}

/**
 * Score data points for anomalies (returns all points with scores)
 */
export async function scoreAnomaly(
  request: AnomalyScoreRequest
): Promise<AnomalyScoreResponse> {
  const response = await apiClient.post<AnomalyScoreResponse>(
    '/ml/anomaly/score',
    request
  )
  return response.data
}

/**
 * Detect anomalies (returns only anomalous points)
 */
export async function detectAnomaly(
  request: AnomalyScoreRequest
): Promise<AnomalyScoreResponse> {
  const response = await apiClient.post<AnomalyScoreResponse>(
    '/ml/anomaly/detect',
    request
  )
  return response.data
}

/**
 * Save a trained anomaly model to disk
 */
export async function saveAnomaly(
  request: AnomalySaveRequest
): Promise<AnomalySaveResponse> {
  const response = await apiClient.post<AnomalySaveResponse>(
    '/ml/anomaly/save',
    request
  )
  return response.data
}

/**
 * Load a pre-trained anomaly model from disk
 */
export async function loadAnomaly(
  request: AnomalyLoadRequest
): Promise<AnomalyLoadResponse> {
  const response = await apiClient.post<AnomalyLoadResponse>(
    '/ml/anomaly/load',
    request
  )
  return response.data
}

/**
 * List all saved anomaly models
 */
export async function listAnomalyModels(): Promise<AnomalyListModelsResponse> {
  const response = await apiClient.get<AnomalyListModelsResponse>('/ml/anomaly/models')
  return response.data
}

/**
 * Delete a saved anomaly model
 */
export async function deleteAnomalyModel(
  filename: string
): Promise<MLDeleteResponse> {
  const response = await apiClient.delete<MLDeleteResponse>(
    `/ml/anomaly/models/${encodeURIComponent(filename)}`
  )
  return response.data
}

// =============================================================================
// Streaming Anomaly Detection Endpoints
// =============================================================================

/**
 * Stream data through an anomaly detector (auto-creates if needed)
 * Implements the Tick-Tock pattern: fast inference with auto-rolling retraining
 */
export async function streamAnomaly(
  request: StreamingAnomalyRequest
): Promise<StreamingAnomalyResponse> {
  const response = await apiClient.post<StreamingAnomalyResponse>(
    '/ml/anomaly/stream',
    request
  )
  return response.data
}

/**
 * List all active streaming detectors
 */
export async function listStreamingDetectors(): Promise<StreamingDetectorListResponse> {
  const response = await apiClient.get<StreamingDetectorListResponse>(
    '/ml/anomaly/stream/detectors'
  )
  return response.data
}

/**
 * Get stats for a specific streaming detector
 */
export async function getStreamingDetector(
  modelId: string
): Promise<StreamingDetectorStats> {
  const response = await apiClient.get<StreamingDetectorStats>(
    `/ml/anomaly/stream/${encodeURIComponent(modelId)}`
  )
  return response.data
}

/**
 * Delete a streaming detector
 */
export async function deleteStreamingDetector(
  modelId: string
): Promise<MLDeleteResponse> {
  const response = await apiClient.delete<MLDeleteResponse>(
    `/ml/anomaly/stream/${encodeURIComponent(modelId)}`
  )
  return response.data
}

/**
 * Reset a streaming detector (clears data, restarts cold start)
 */
export async function resetStreamingDetector(
  modelId: string
): Promise<StreamingDetectorStats> {
  const response = await apiClient.post<StreamingDetectorStats>(
    `/ml/anomaly/stream/${encodeURIComponent(modelId)}/reset`
  )
  return response.data
}

// =============================================================================
// Document Scanning Endpoints
// =============================================================================

/**
 * Scan a document (image or PDF) and extract text using OCR
 * Uses the existing /v1/vision/ocr endpoint
 */
export async function scanDocument(
  file: File,
  options: {
    model?: string
    languages?: string
    return_boxes?: boolean
    parse_by_page?: boolean
  } = {}
): Promise<DocumentScanningResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('model', options.model || 'surya')
  formData.append('languages', options.languages || 'en')
  formData.append('return_boxes', String(options.return_boxes || false))
  formData.append('parse_by_page', String(options.parse_by_page || false))

  const response = await apiClient.post<DocumentScanningResponse>(
    '/vision/ocr',
    formData,
    {
      // Must explicitly clear Content-Type so axios sets multipart/form-data with correct boundary
      // The apiClient default 'application/json' header would otherwise interfere
      headers: {
        'Content-Type': undefined,
      },
      // OCR can take a while, especially first time when loading models
      timeout: 300000, // 5 minutes
    }
  )
  return response.data
}

// =============================================================================
// Encoder Endpoints (Embeddings & Reranking)
// =============================================================================

// Universal Runtime URL - calls directly to runtime for encoder operations
// Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
const UNIVERSAL_RUNTIME_URL =
  import.meta.env.VITE_UNIVERSAL_RUNTIME_URL || 'http://127.0.0.1:11540'

/**
 * Generate embeddings for texts
 * Calls Universal Runtime directly at /v1/embeddings
 */
export async function createEmbeddings(
  request: EmbeddingRequest
): Promise<EmbeddingResponse> {
  const response = await fetch(`${UNIVERSAL_RUNTIME_URL}/v1/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

/**
 * Rerank documents based on query relevance
 * Calls Universal Runtime directly at /v1/rerank
 */
export async function rerankDocuments(
  request: RerankRequest
): Promise<RerankResponse> {
  const response = await fetch(`${UNIVERSAL_RUNTIME_URL}/v1/rerank`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

// =============================================================================
// Default Export
// =============================================================================

export default {
  // Health
  checkMLHealth,
  // Classifier
  fitClassifier,
  predictClassifier,
  saveClassifier,
  loadClassifier,
  listClassifierModels,
  deleteClassifierModel,
  // Anomaly
  fitAnomaly,
  scoreAnomaly,
  detectAnomaly,
  saveAnomaly,
  loadAnomaly,
  listAnomalyModels,
  deleteAnomalyModel,
  // Streaming Anomaly
  streamAnomaly,
  listStreamingDetectors,
  getStreamingDetector,
  deleteStreamingDetector,
  resetStreamingDetector,
  // Document Scanning
  scanDocument,
  // Encoder
  createEmbeddings,
  rerankDocuments,
}
