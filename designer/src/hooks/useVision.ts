/**
 * React Query hooks for Vision API operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import visionService from '../api/visionService'
import type {
  DetectRequest,
  ClassifyRequest,
  DetectClassifyRequest,
  TrainRequest,
  StreamStartRequest,
  StreamFrameRequest,
  StreamStopRequest,
  SaveModelRequest,
  LoadModelRequest,
  ExportModelRequest,
  ReviewDecideRequest,
} from '../types/vision'

// =============================================================================
// Query Keys
// =============================================================================

export const visionKeys = {
  all: ['vision'] as const,
  models: () => [...visionKeys.all, 'models'] as const,
  modelList: () => [...visionKeys.models(), 'list'] as const,
  training: () => [...visionKeys.all, 'training'] as const,
  trainingJob: (jobId: string) => [...visionKeys.training(), jobId] as const,
  streaming: () => [...visionKeys.all, 'streaming'] as const,
  streamSessions: () => [...visionKeys.streaming(), 'sessions'] as const,
  review: () => [...visionKeys.all, 'review'] as const,
  reviewPending: (page: number, pageSize: number) => [...visionKeys.review(), 'pending', page, pageSize] as const,
}

// =============================================================================
// Model Queries
// =============================================================================

export function useVisionModels(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: visionKeys.modelList(),
    queryFn: () => visionService.listModels(),
    enabled: options?.enabled !== false,
    staleTime: 5_000,
    refetchOnMount: 'always' as const,
  })
}

// =============================================================================
// Training Queries
// =============================================================================

export function useTrainingJobStatus(
  jobId: string | null,
  options?: { enabled?: boolean; refetchInterval?: number }
) {
  return useQuery({
    queryKey: visionKeys.trainingJob(jobId!),
    queryFn: () => visionService.getTrainingStatus(jobId!),
    enabled: !!jobId && options?.enabled !== false,
    refetchInterval: query => {
      const data = query.state.data
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false
      }
      return options?.refetchInterval || 2000
    },
    refetchIntervalInBackground: true,
    staleTime: 0,
  })
}

// =============================================================================
// Stream Queries
// =============================================================================

export function useStreamSessions(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: visionKeys.streamSessions(),
    queryFn: async () => {
      try {
        return await visionService.listStreamSessions()
      } catch {
        // Endpoint may not exist on older runtimes — return empty
        return { sessions: [] }
      }
    },
    enabled: options?.enabled !== false,
    refetchInterval: 5_000,
    staleTime: 2_000,
    retry: false,
  })
}

// =============================================================================
// Review Queries
// =============================================================================

export function usePendingReviews(
  page: number,
  pageSize = 10,
  options?: { enabled?: boolean }
) {
  return useQuery({
    queryKey: visionKeys.reviewPending(page, pageSize),
    queryFn: () => visionService.getPendingReviews(page, pageSize),
    enabled: options?.enabled !== false,
    staleTime: 5_000,
  })
}

// =============================================================================
// Detection Mutations
// =============================================================================

export function useDetect() {
  return useMutation({
    mutationFn: (request: DetectRequest) => visionService.detect(request),
  })
}

export function useClassify() {
  return useMutation({
    mutationFn: (request: ClassifyRequest) => visionService.classify(request),
  })
}

export function useDetectClassify() {
  return useMutation({
    mutationFn: (request: DetectClassifyRequest) =>
      visionService.detectClassify(request),
  })
}

// =============================================================================
// Training Mutations
// =============================================================================

export function useStartTraining() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: TrainRequest) => visionService.train(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.modelList() })
    },
  })
}

// =============================================================================
// Streaming Mutations
// =============================================================================

export function useStreamStart() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: StreamStartRequest) => visionService.streamStart(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.streamSessions() })
    },
  })
}

export function useStreamFrame() {
  return useMutation({
    mutationFn: (request: StreamFrameRequest) => visionService.streamFrame(request),
  })
}

export function useStreamStop() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: StreamStopRequest) => visionService.streamStop(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.streamSessions() })
    },
  })
}

// =============================================================================
// Model Mutations
// =============================================================================

export function useSaveVisionModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: SaveModelRequest) => visionService.saveModel(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.modelList() })
    },
  })
}

export function useLoadVisionModel() {
  return useMutation({
    mutationFn: (request: LoadModelRequest) => visionService.loadModel(request),
  })
}

export function useExportVisionModel() {
  return useMutation({
    mutationFn: (request: ExportModelRequest) => visionService.exportModel(request),
  })
}

export function useDeleteVisionModel() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => visionService.deleteModel(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.modelList() })
    },
  })
}

// =============================================================================
// Review Mutations
// =============================================================================

export function useReviewDecision() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: ReviewDecideRequest) =>
      visionService.submitReviewDecision(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: visionKeys.review() })
    },
  })
}

// =============================================================================
// Sample Data
// =============================================================================

export function useSampleDataStatus() {
  return useQuery({
    queryKey: [...visionKeys.all, 'sampleData', 'status'],
    queryFn: () => visionService.getSampleDataStatus(),
    retry: false,
    staleTime: 30_000,
  })
}

export function useCloneSampleData() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => visionService.cloneSampleData(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [...visionKeys.all, 'sampleData'] })
    },
  })
}
