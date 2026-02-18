/**
 * Vision Service - API client for vision detection, classification, training, and streaming
 * Endpoints proxy to Universal Runtime via LlamaFarm server
 */

import { runtimeClient } from './client'
import type {
  DetectRequest,
  DetectResponse,
  ClassifyRequest,
  ClassifyResponse,
  DetectClassifyRequest,
  DetectClassifyResponse,
  TrainRequest,
  TrainResponse,
  TrainingJobStatus,
  StreamStartRequest,
  StreamStartResponse,
  StreamFrameRequest,
  StreamFrameResponse,
  StreamStopRequest,
  StreamStopResponse,
  StreamSessionsResponse,
  VisionModelsResponse,
  SaveModelRequest,
  LoadModelRequest,
  ExportModelRequest,
  PendingReviewResponse,
  ReviewDecideRequest,
  ReviewDecideResponse,
} from '../types/vision'

// =============================================================================
// Detection
// =============================================================================

export async function detect(request: DetectRequest): Promise<DetectResponse> {
  const response = await runtimeClient.post<DetectResponse>('/v1/vision/detect', request)
  return response.data
}

export async function classify(request: ClassifyRequest): Promise<ClassifyResponse> {
  const response = await runtimeClient.post<ClassifyResponse>('/v1/vision/classify', request)
  return response.data
}

export async function detectClassify(
  request: DetectClassifyRequest
): Promise<DetectClassifyResponse> {
  const response = await runtimeClient.post<DetectClassifyResponse>(
    '/v1/vision/detect_classify',
    request
  )
  return response.data
}

// =============================================================================
// Training
// =============================================================================

export async function train(request: TrainRequest): Promise<TrainResponse> {
  const response = await runtimeClient.post<TrainResponse>('/v1/vision/train', request)
  return response.data
}

export async function getTrainingStatus(jobId: string): Promise<TrainingJobStatus> {
  const response = await runtimeClient.get<TrainingJobStatus>(
    `/v1/vision/train/${encodeURIComponent(jobId)}`
  )
  return response.data
}

// =============================================================================
// Streaming
// =============================================================================

export async function streamStart(request: StreamStartRequest): Promise<StreamStartResponse> {
  const response = await runtimeClient.post<StreamStartResponse>(
    '/v1/vision/stream/start',
    request
  )
  return response.data
}

export async function streamFrame(request: StreamFrameRequest): Promise<StreamFrameResponse> {
  const response = await runtimeClient.post<StreamFrameResponse>(
    '/v1/vision/stream/frame',
    request
  )
  return response.data
}

export async function streamStop(request: StreamStopRequest): Promise<StreamStopResponse> {
  const response = await runtimeClient.post<StreamStopResponse>(
    '/v1/vision/stream/stop',
    request
  )
  return response.data
}

export async function listStreamSessions(): Promise<StreamSessionsResponse> {
  const response = await runtimeClient.get<StreamSessionsResponse>(
    '/v1/vision/stream/sessions'
  )
  return response.data
}

// =============================================================================
// Models
// =============================================================================

export async function listModels(): Promise<VisionModelsResponse> {
  const response = await runtimeClient.get<VisionModelsResponse>('/v1/vision/models')
  return response.data
}

export async function saveModel(request: SaveModelRequest): Promise<{ success: boolean }> {
  const response = await runtimeClient.post('/v1/vision/models/save', request)
  return response.data
}

export async function loadModel(request: LoadModelRequest): Promise<{ success: boolean }> {
  const response = await runtimeClient.post('/v1/vision/models/load', request)
  return response.data
}

export async function exportModel(request: ExportModelRequest): Promise<Blob> {
  const response = await runtimeClient.post('/v1/vision/models/export', request, {
    responseType: 'blob',
  })
  return response.data
}

// =============================================================================
// Review
// =============================================================================

export async function getPendingReviews(
  page = 1,
  pageSize = 10
): Promise<PendingReviewResponse> {
  const response = await runtimeClient.get<PendingReviewResponse>(
    '/v1/vision/review/pending',
    { params: { page, page_size: pageSize } }
  )
  return response.data
}

export async function submitReviewDecision(
  request: ReviewDecideRequest
): Promise<ReviewDecideResponse> {
  const response = await runtimeClient.post<ReviewDecideResponse>(
    '/v1/vision/review/decide',
    request
  )
  return response.data
}

// =============================================================================
// Default Export
// =============================================================================

export default {
  detect,
  classify,
  detectClassify,
  train,
  getTrainingStatus,
  streamStart,
  streamFrame,
  streamStop,
  listStreamSessions,
  listModels,
  saveModel,
  loadModel,
  exportModel,
  getPendingReviews,
  submitReviewDecision,
}
