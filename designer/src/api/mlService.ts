/**
 * ML Service - API client for classifier, anomaly detection, and router endpoints
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
  // Router types
  RouterTrainRequest,
  RouterTrainResponse,
  RouterRouteRequest,
  RouterRouteResponse,
  RouterLoadRequest,
  RouterLoadResponse,
  RouterListModelsResponse,
  RouterGenerateDataRequest,
  RouterGenerateDataResponse,
  RouterBatchGenerateDataResponse,
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
// Router Endpoints
// =============================================================================

/**
 * Train a semantic router with routes and utterances
 */
export async function trainRouter(
  request: RouterTrainRequest
): Promise<RouterTrainResponse> {
  const response = await apiClient.post<RouterTrainResponse>(
    '/ml/router/train',
    request
  )
  return response.data
}

/**
 * Route a query using a trained router
 */
export async function routeQuery(
  request: RouterRouteRequest
): Promise<RouterRouteResponse> {
  const response = await apiClient.post<RouterRouteResponse>(
    '/ml/router/route',
    request
  )
  return response.data
}

/**
 * Load a saved router into memory
 */
export async function loadRouter(
  request: RouterLoadRequest
): Promise<RouterLoadResponse> {
  const response = await apiClient.post<RouterLoadResponse>(
    '/ml/router/load',
    request
  )
  return response.data
}

/**
 * List all saved router models
 */
export async function listRouterModels(): Promise<RouterListModelsResponse> {
  const response = await apiClient.get<RouterListModelsResponse>('/ml/router/models')
  return response.data
}

/**
 * Delete a saved router model
 */
export async function deleteRouterModel(
  modelName: string
): Promise<MLDeleteResponse> {
  const response = await apiClient.delete<MLDeleteResponse>(
    `/ml/router/models/${encodeURIComponent(modelName)}`
  )
  return response.data
}

/**
 * Generate synthetic training data for router routes
 */
export async function generateRouterData(
  request: RouterGenerateDataRequest
): Promise<RouterGenerateDataResponse | RouterBatchGenerateDataResponse> {
  const response = await apiClient.post<RouterGenerateDataResponse | RouterBatchGenerateDataResponse>(
    '/ml/router/generate-data',
    request
  )
  return response.data
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
  // Router
  trainRouter,
  routeQuery,
  loadRouter,
  listRouterModels,
  deleteRouterModel,
  generateRouterData,
}
