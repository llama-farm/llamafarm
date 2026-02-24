/**
 * Vision Service - API client for vision detection, classification, training, and streaming
 * Endpoints proxy to Universal Runtime via LlamaFarm server
 */

import axios from 'axios'

// Vision endpoints go through the Vite dev proxy (/v1/vision/* → runtime:11540)
// rather than direct to runtime, avoiding CORS issues in development
const visionClient = axios.create({
  baseURL: '',  // relative — uses current origin, proxied by Vite
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})
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
  const response = await visionClient.post<DetectResponse>('/v1/vision/detect', request)
  return response.data
}

export async function classify(request: ClassifyRequest): Promise<ClassifyResponse> {
  const response = await visionClient.post<ClassifyResponse>('/v1/vision/classify', request)
  return response.data
}

export async function detectClassify(
  request: DetectClassifyRequest
): Promise<DetectClassifyResponse> {
  const response = await visionClient.post<DetectClassifyResponse>(
    '/v1/vision/detect_classify',
    request
  )
  return response.data
}

// =============================================================================
// Training
// =============================================================================

export async function train(request: TrainRequest): Promise<TrainResponse> {
  const response = await visionClient.post<TrainResponse>('/v1/vision/train', request)
  return response.data
}

export async function getTrainingStatus(jobId: string): Promise<TrainingJobStatus> {
  const response = await visionClient.get<TrainingJobStatus>(
    `/v1/vision/train/${encodeURIComponent(jobId)}`
  )
  return response.data
}

// =============================================================================
// Streaming
// =============================================================================

export async function streamStart(request: StreamStartRequest): Promise<StreamStartResponse> {
  const response = await visionClient.post<StreamStartResponse>(
    '/v1/vision/stream/start',
    request
  )
  return response.data
}

export async function streamFrame(request: StreamFrameRequest): Promise<StreamFrameResponse> {
  const response = await visionClient.post<StreamFrameResponse>(
    '/v1/vision/stream/frame',
    request
  )
  return response.data
}

export async function streamStop(request: StreamStopRequest): Promise<StreamStopResponse> {
  const response = await visionClient.post<StreamStopResponse>(
    '/v1/vision/stream/stop',
    request
  )
  return response.data
}

export async function listStreamSessions(): Promise<StreamSessionsResponse> {
  const response = await visionClient.get<StreamSessionsResponse>(
    '/v1/vision/stream/sessions'
  )
  return response.data
}

// =============================================================================
// Models
// =============================================================================

export async function listModels(): Promise<VisionModelsResponse> {
  const response = await visionClient.get<VisionModelsResponse>('/v1/vision/models')
  return response.data
}

export async function saveModel(request: SaveModelRequest): Promise<{ success: boolean }> {
  const response = await visionClient.post('/v1/vision/models/save', request)
  return response.data
}

export async function loadModel(request: LoadModelRequest): Promise<{ success: boolean }> {
  const response = await visionClient.post('/v1/vision/models/load', request)
  return response.data
}

export async function exportModel(request: ExportModelRequest): Promise<Blob> {
  const response = await visionClient.post('/v1/vision/models/export', request, {
    responseType: 'blob',
  })
  return response.data
}

export async function deleteModel(name: string): Promise<{ deleted: boolean }> {
  const response = await visionClient.delete(`/v1/vision/models/${encodeURIComponent(name)}`)
  return response.data
}

// =============================================================================
// Review
// =============================================================================

export async function getPendingReviews(
  page = 1,
  pageSize = 10
): Promise<PendingReviewResponse> {
  const response = await visionClient.get<PendingReviewResponse>(
    '/v1/vision/review/pending',
    { params: { page, page_size: pageSize } }
  )
  return response.data
}

export async function submitReviewDecision(
  request: ReviewDecideRequest
): Promise<ReviewDecideResponse> {
  const response = await visionClient.post<ReviewDecideResponse>(
    '/v1/vision/review/decide',
    request
  )
  return response.data
}

// =============================================================================
// Sample Data
// =============================================================================

export interface SampleDataStatus {
  installed: boolean
  path: string
  categories: string[]
}

export interface CloneResponse {
  success: boolean
  path: string
  message: string
}

export async function getSampleDataStatus(): Promise<SampleDataStatus> {
  const response = await visionClient.get<SampleDataStatus>('/v1/vision/sample-data/status')
  return response.data
}

export async function cloneSampleData(): Promise<CloneResponse> {
  const response = await visionClient.post<CloneResponse>('/v1/vision/sample-data/clone')
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
  deleteModel,
  getPendingReviews,
  submitReviewDecision,
  getSampleDataStatus,
  cloneSampleData,
}
