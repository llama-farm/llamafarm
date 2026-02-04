/**
 * React Query hooks for ML model operations (classifier and anomaly detection)
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import mlService from '../api/mlService'
import type {
  ClassifierFitRequest,
  ClassifierPredictRequest,
  ClassifierSaveRequest,
  ClassifierLoadRequest,
  AnomalyFitRequest,
  AnomalyScoreRequest,
  AnomalySaveRequest,
  AnomalyLoadRequest,
  StreamingAnomalyRequest,
  EmbeddingRequest,
  RerankRequest,
} from '../types/ml'

// =============================================================================
// Query Keys
// =============================================================================

export const mlModelKeys = {
  all: ['ml-models'] as const,
  health: () => [...mlModelKeys.all, 'health'] as const,
  // Classifier keys
  classifiers: () => [...mlModelKeys.all, 'classifiers'] as const,
  classifierList: () => [...mlModelKeys.classifiers(), 'list'] as const,
  // Anomaly keys
  anomalies: () => [...mlModelKeys.all, 'anomalies'] as const,
  anomalyList: () => [...mlModelKeys.anomalies(), 'list'] as const,
  // Streaming anomaly keys
  streamingDetectors: () => [...mlModelKeys.all, 'streaming-detectors'] as const,
  streamingDetectorList: () => [...mlModelKeys.streamingDetectors(), 'list'] as const,
  streamingDetector: (modelId: string) =>
    [...mlModelKeys.streamingDetectors(), 'detail', modelId] as const,
}

// =============================================================================
// Health Check
// =============================================================================

/**
 * Check ML service health
 */
export function useMLHealth(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: mlModelKeys.health(),
    queryFn: () => mlService.checkMLHealth(),
    enabled: options?.enabled !== false,
    staleTime: 30_000, // 30 seconds
    retry: 1,
  })
}

// =============================================================================
// Classifier Queries
// =============================================================================

/**
 * List all saved classifier models
 */
export function useListClassifierModels(options?: {
  enabled?: boolean
  staleTime?: number
}) {
  return useQuery({
    queryKey: mlModelKeys.classifierList(),
    queryFn: () => mlService.listClassifierModels(),
    enabled: options?.enabled !== false,
    staleTime: options?.staleTime ?? 5_000, // 5 seconds - short to catch new models quickly
    refetchOnMount: 'always', // Always refetch when component mounts
  })
}

// =============================================================================
// Classifier Mutations
// =============================================================================

/**
 * Train a text classifier
 */
export function useTrainClassifier() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ClassifierFitRequest) =>
      mlService.fitClassifier(request),
    onSuccess: () => {
      // Invalidate classifier list to show new model
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.classifierList(),
      })
    },
  })
}

/**
 * Classify texts using a trained model
 */
export function usePredictClassifier() {
  return useMutation({
    mutationFn: (request: ClassifierPredictRequest) =>
      mlService.predictClassifier(request),
  })
}

/**
 * Save a trained classifier to disk
 */
export function useSaveClassifier() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: ClassifierSaveRequest) =>
      mlService.saveClassifier(request),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.classifierList(),
      })
    },
  })
}

/**
 * Load a classifier from disk
 */
export function useLoadClassifier() {
  return useMutation({
    mutationFn: (request: ClassifierLoadRequest) =>
      mlService.loadClassifier(request),
  })
}

/**
 * Delete a saved classifier model
 */
export function useDeleteClassifierModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (modelName: string) =>
      mlService.deleteClassifierModel(modelName),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.classifierList(),
      })
    },
  })
}

// =============================================================================
// Anomaly Queries
// =============================================================================

/**
 * List all saved anomaly models
 */
export function useListAnomalyModels(options?: {
  enabled?: boolean
  staleTime?: number
}) {
  return useQuery({
    queryKey: mlModelKeys.anomalyList(),
    queryFn: () => mlService.listAnomalyModels(),
    enabled: options?.enabled !== false,
    staleTime: options?.staleTime ?? 5_000, // 5 seconds - short to catch new models quickly
    refetchOnMount: 'always', // Always refetch when component mounts
  })
}

// =============================================================================
// Anomaly Mutations
// =============================================================================

/**
 * Train an anomaly detection model
 */
export function useTrainAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: AnomalyFitRequest) => mlService.fitAnomaly(request),
    onSuccess: () => {
      // Invalidate anomaly list to show new model
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.anomalyList(),
      })
    },
  })
}

/**
 * Score data for anomalies
 */
export function useScoreAnomaly() {
  return useMutation({
    mutationFn: (request: AnomalyScoreRequest) =>
      mlService.scoreAnomaly(request),
  })
}

/**
 * Save a trained anomaly model to disk
 */
export function useSaveAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: AnomalySaveRequest) => mlService.saveAnomaly(request),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.anomalyList(),
      })
    },
  })
}

/**
 * Load an anomaly model from disk
 */
export function useLoadAnomaly() {
  return useMutation({
    mutationFn: (request: AnomalyLoadRequest) => mlService.loadAnomaly(request),
  })
}

/**
 * Delete a saved anomaly model
 */
export function useDeleteAnomalyModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (filename: string) => mlService.deleteAnomalyModel(filename),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.anomalyList(),
      })
    },
  })
}

// =============================================================================
// Streaming Anomaly Detection Queries
// =============================================================================

/**
 * List all active streaming detectors
 * Polls every 5 seconds when enabled
 */
export function useListStreamingDetectors(options?: {
  enabled?: boolean
  refetchInterval?: number
}) {
  return useQuery({
    queryKey: mlModelKeys.streamingDetectorList(),
    queryFn: () => mlService.listStreamingDetectors(),
    enabled: options?.enabled !== false,
    refetchInterval: options?.refetchInterval ?? 5_000, // 5 seconds
    staleTime: 2_000, // 2 seconds
  })
}

/**
 * Get stats for a specific streaming detector
 * Polls every 2 seconds when enabled (faster for live status updates)
 */
export function useStreamingDetector(
  modelId: string,
  options?: {
    enabled?: boolean
    refetchInterval?: number
  }
) {
  return useQuery({
    queryKey: mlModelKeys.streamingDetector(modelId),
    queryFn: () => mlService.getStreamingDetector(modelId),
    enabled: options?.enabled !== false && !!modelId,
    refetchInterval: options?.refetchInterval ?? 2_000, // 2 seconds
    staleTime: 1_000, // 1 second
    retry: false, // Don't retry if detector doesn't exist
  })
}

// =============================================================================
// Streaming Anomaly Detection Mutations
// =============================================================================

/**
 * Stream data through an anomaly detector
 * Auto-creates detector if it doesn't exist
 */
export function useStreamAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: StreamingAnomalyRequest) =>
      mlService.streamAnomaly(request),
    onSuccess: (_data, variables) => {
      // Invalidate detector list and specific detector stats
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.streamingDetectorList(),
      })
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.streamingDetector(variables.model),
      })
    },
  })
}

/**
 * Delete a streaming detector
 */
export function useDeleteStreamingDetector() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (modelId: string) => mlService.deleteStreamingDetector(modelId),
    onSuccess: (_data, modelId) => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.streamingDetectorList(),
      })
      queryClient.removeQueries({
        queryKey: mlModelKeys.streamingDetector(modelId),
      })
    },
  })
}

/**
 * Reset a streaming detector (clears data, restarts cold start)
 */
export function useResetStreamingDetector() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (modelId: string) => mlService.resetStreamingDetector(modelId),
    onSuccess: (_data, modelId) => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.streamingDetector(modelId),
      })
    },
  })
}

// =============================================================================
// Combined Hooks
// =============================================================================

/**
 * Train a classifier (fit auto-saves the model)
 */
export function useTrainAndSaveClassifier() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: ClassifierFitRequest) => {
      // Fit the model - this auto-saves to disk
      const fitResult = await mlService.fitClassifier(request)

      // No separate save call needed - fit endpoint auto-saves
      return { fitResult }
    },
    onSuccess: () => {
      // Invalidate and force refetch to ensure models list is up to date
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.classifierList(),
        refetchType: 'all',
      })
    },
  })
}

/**
 * Train an anomaly detector (fit auto-saves the model)
 */
export function useTrainAndSaveAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: AnomalyFitRequest) => {
      // Fit the model - this auto-saves to disk
      const fitResult = await mlService.fitAnomaly(request)

      // No separate save call needed - fit endpoint auto-saves
      return { fitResult }
    },
    onSuccess: () => {
      // Invalidate and force refetch to ensure models list is up to date
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.anomalyList(),
        refetchType: 'all',
      })
    },
  })
}

// =============================================================================
// Document Scanning Mutations
// =============================================================================

/**
 * Scan a document (image or PDF) and extract text using OCR
 */
export function useScanDocument() {
  return useMutation({
    mutationFn: ({
      file,
      model,
      languages,
      returnBoxes,
      parseByPage,
    }: {
      file: File
      model?: string
      languages?: string
      returnBoxes?: boolean
      parseByPage?: boolean
    }) =>
      mlService.scanDocument(file, {
        model,
        languages,
        return_boxes: returnBoxes,
        parse_by_page: parseByPage,
      }),
  })
}

// =============================================================================
// Encoder Mutations (Embeddings & Reranking)
// =============================================================================

/**
 * Generate embeddings for texts
 */
export function useCreateEmbeddings() {
  return useMutation({
    mutationFn: (request: EmbeddingRequest) => mlService.createEmbeddings(request),
  })
}

/**
 * Rerank documents based on query relevance
 */
export function useRerankDocuments() {
  return useMutation({
    mutationFn: (request: RerankRequest) => mlService.rerankDocuments(request),
  })
}

// =============================================================================
// Default Export
// =============================================================================

export default {
  // Keys
  mlModelKeys,
  // Health
  useMLHealth,
  // Classifier queries
  useListClassifierModels,
  // Classifier mutations
  useTrainClassifier,
  usePredictClassifier,
  useSaveClassifier,
  useLoadClassifier,
  useDeleteClassifierModel,
  useTrainAndSaveClassifier,
  // Anomaly queries
  useListAnomalyModels,
  // Anomaly mutations
  useTrainAnomaly,
  useScoreAnomaly,
  useSaveAnomaly,
  useLoadAnomaly,
  useDeleteAnomalyModel,
  useTrainAndSaveAnomaly,
  // Streaming anomaly queries
  useListStreamingDetectors,
  useStreamingDetector,
  // Streaming anomaly mutations
  useStreamAnomaly,
  useDeleteStreamingDetector,
  useResetStreamingDetector,
  // Document Scanning mutations
  useScanDocument,
  // Encoder mutations
  useCreateEmbeddings,
  useRerankDocuments,
}
