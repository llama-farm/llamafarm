/**
 * Vision API Types
 * Matches the backend endpoints at /v1/vision/*
 */

// =============================================================================
// Detection
// =============================================================================

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  box: BoundingBox
  class_name: string
  confidence: number
}

export interface DetectRequest {
  image: string // base64
  model?: string
  confidence_threshold?: number
  classes?: string[]
}

export interface DetectResponse {
  detections: Detection[]
}

// =============================================================================
// Classification
// =============================================================================

export interface ClassifyRequest {
  image: string // base64
  model?: string
  classes?: string[]
  top_k?: number
}

export interface ClassifyResponse {
  class_name: string
  confidence: number
  all_scores: Record<string, number>
}

// =============================================================================
// Detect + Classify
// =============================================================================

export interface DetectClassifyRequest {
  image: string // base64
  detect_model?: string
  classify_model?: string
  confidence_threshold?: number
  detect_classes?: string[]
  classify_classes?: string[]
}

export interface DetectClassifyResult {
  detection: Detection
  classification: ClassifyResponse
}

export interface DetectClassifyResponse {
  results: DetectClassifyResult[]
}

// =============================================================================
// Training
// =============================================================================

export interface TrainingConfig {
  epochs?: number
  batch_size?: number
  learning_rate?: number
}

export interface TrainRequest {
  model: string
  dataset: string
  task: string
  base_model?: string
  config?: TrainingConfig
}

export interface TrainResponse {
  job_id: string
  status: string
}

export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface TrainingJobStatus {
  job_id: string
  status: TrainingStatus
  progress?: number
  epoch?: number
  total_epochs?: number
  loss?: number
  message?: string
}

// =============================================================================
// Streaming
// =============================================================================

export interface StreamConfig {
  model?: string
  confidence_threshold?: number
  classes?: string[]
}

export interface StreamStartRequest {
  config?: StreamConfig
  target_fps?: number
  cooldown_seconds?: number
}

export interface StreamStartResponse {
  session_id: string
}

export interface StreamFrameRequest {
  session_id: string
  image: string // base64
}

export interface StreamFrameResponse {
  detections: Detection[]
}

export interface StreamStopRequest {
  session_id: string
}

export interface StreamStopResponse {
  session_id: string
  frames_processed: number
  avg_fps: number
  duration_seconds: number
}

export interface StreamSession {
  session_id: string
  created_at: string
  frames_processed: number
  target_fps: number
}

export interface StreamSessionsResponse {
  sessions: StreamSession[]
}

// =============================================================================
// Models
// =============================================================================

export interface VisionModel {
  name: string
  task?: string
  created_at?: string
  size?: number
  size_mb?: number
  versions?: number
  has_current?: boolean
  description?: string
}

export interface VisionModelsResponse {
  models: VisionModel[]
}

export interface SaveModelRequest {
  name: string
  description?: string
}

export interface LoadModelRequest {
  name: string
}

export interface ExportModelRequest {
  name: string
  format?: string
}

// =============================================================================
// Review
// =============================================================================

export interface ReviewItem {
  id: string
  image: string // base64 crop
  detection: Detection
  model: string
  timestamp: string
}

export interface PendingReviewResponse {
  items: ReviewItem[]
  total: number
  page: number
  page_size: number
}

export type ReviewDecision = 'correct' | 'wrong' | 'skip'

export interface ReviewDecideRequest {
  id: string
  decision: ReviewDecision
  corrected_class?: string
}

export interface ReviewDecideResponse {
  success: boolean
}
