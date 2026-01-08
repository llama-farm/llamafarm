/**
 * React Query hooks for ML model operations (classifier, anomaly detection, and router)
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
  RouterTrainRequest,
  RouterRouteRequest,
  RouterLoadRequest,
  RouterGenerateDataRequest,
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
  // Router keys
  routers: () => [...mlModelKeys.all, 'routers'] as const,
  routerList: () => [...mlModelKeys.routers(), 'list'] as const,
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
// Combined Hooks
// =============================================================================

/**
 * Train and save a classifier in one operation
 */
export function useTrainAndSaveClassifier() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: ClassifierFitRequest) => {
      // First fit the model
      const fitResult = await mlService.fitClassifier(request)

      // Then save it to disk (pass description to save endpoint)
      const saveResult = await mlService.saveClassifier({
        model: fitResult.versioned_name,
        description: request.description,
      })

      return { fitResult, saveResult }
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
 * Train and save an anomaly detector in one operation
 */
export function useTrainAndSaveAnomaly() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: async (request: AnomalyFitRequest) => {
      // First fit the model
      const fitResult = await mlService.fitAnomaly(request)

      // Then save it to disk (pass description to save endpoint)
      const saveResult = await mlService.saveAnomaly({
        model: fitResult.versioned_name,
        backend: request.backend || 'isolation_forest',
        description: request.description,
      })

      return { fitResult, saveResult }
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
// Router Queries
// =============================================================================

/**
 * List all saved router models
 */
export function useListRouterModels(options?: {
  enabled?: boolean
  staleTime?: number
}) {
  return useQuery({
    queryKey: mlModelKeys.routerList(),
    queryFn: () => mlService.listRouterModels(),
    enabled: options?.enabled !== false,
    staleTime: options?.staleTime ?? 5_000, // 5 seconds - short to catch new models quickly
    refetchOnMount: 'always', // Always refetch when component mounts
  })
}

// =============================================================================
// Router Mutations
// =============================================================================

/**
 * Train a semantic router
 */
export function useTrainRouter() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: RouterTrainRequest) => mlService.trainRouter(request),
    onSuccess: () => {
      // Invalidate router list to show new model
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.routerList(),
      })
    },
  })
}

/**
 * Route a query using a trained router
 */
export function useRouteQuery() {
  return useMutation({
    mutationFn: (request: RouterRouteRequest) => mlService.routeQuery(request),
  })
}

/**
 * Load a saved router into memory
 */
export function useLoadRouter() {
  return useMutation({
    mutationFn: (request: RouterLoadRequest) => mlService.loadRouter(request),
  })
}

/**
 * Delete a saved router model
 */
export function useDeleteRouterModel() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (modelName: string) => mlService.deleteRouterModel(modelName),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: mlModelKeys.routerList(),
      })
    },
  })
}

/**
 * Generate synthetic training data for router routes
 */
export function useGenerateRouterData() {
  return useMutation({
    mutationFn: (request: RouterGenerateDataRequest) =>
      mlService.generateRouterData(request),
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
  // Router queries
  useListRouterModels,
  // Router mutations
  useTrainRouter,
  useRouteQuery,
  useLoadRouter,
  useDeleteRouterModel,
  useGenerateRouterData,
}
