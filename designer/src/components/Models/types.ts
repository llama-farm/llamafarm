export type ModelStatus = 'ready' | 'downloading'

export interface InferenceModel {
  id: string
  name: string
  modelIdentifier?: string
  meta: string
  badges: string[]
  isDefault?: boolean
  status?: ModelStatus
  // Cloud model configuration
  provider?: string
  apiKey?: string
  baseUrl?: string
  maxTokens?: number | null
}

// Trained models types
export type TrainedModelType = 'anomaly_detection' | 'classifier'
export type TrainedModelStatus = 'ready' | 'training' | 'failed'

export interface TrainedModelVersion {
  id: string
  version: number
  createdAt: string // ISO date string
  trainingSamples: number
  isActive: boolean
  threshold?: number
  baseModel?: string
  // Full versioned name from API (e.g., "model-name_20251218_143022")
  versionedName?: string
}

export interface TrainedModel {
  id: string
  name: string
  type: TrainedModelType
  status: TrainedModelStatus
  versionCount: number
  lastTrained: string // ISO date string
  description?: string
  versions?: TrainedModelVersion[]
  threshold?: number
  baseModel?: string
  // Anomaly-specific fields
  backend?: string
  // Classifier-specific fields
  labels?: string[]
}

// Test result types
export interface ClassifierTestResult {
  id: string
  input: string
  label: string
  confidence: number
  timestamp: string
}

export interface AnomalyTestResult {
  id: string
  input: string
  isAnomaly: boolean
  score: number
  threshold: number
  timestamp: string
  status?: 'success' | 'error' // undefined means success for backward compatibility
}

