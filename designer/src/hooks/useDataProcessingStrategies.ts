import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useUpdateProject, projectKeys } from './useProjects'

/**
 * Type for a data processing strategy
 */
export type DataProcessingStrategy = {
  name: string
  description?: string
  parsers: Array<{
    type: string
    config?: Record<string, unknown>
    file_include_patterns?: string[]
    priority?: number
  }>
  extractors?: Array<{
    type: string
    config?: Record<string, unknown>
    file_include_patterns?: string[]
    priority?: number
  }>
}

/**
 * Hook to manage data processing strategies in the config
 * Provides mutations for creating, updating, and deleting strategies
 *
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @returns Mutation hooks for strategy operations
 */
export const useDataProcessingStrategies = (
  namespace: string,
  projectId: string
) => {
  const queryClient = useQueryClient()
  const updateProjectMutation = useUpdateProject()

  /**
   * Create a new data processing strategy
   */
  const createStrategy = useMutation({
    mutationFn: async ({
      strategy,
      projectConfig,
    }: {
      strategy: DataProcessingStrategy
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for strategy creation')
      }

      // Get or initialize rag.data_processing_strategies
      const rag = projectConfig.rag || {}
      const strategies = rag.data_processing_strategies || []

      // Check if strategy name already exists
      const exists = strategies.some((s: any) => s.name === strategy.name)
      if (exists) {
        throw new Error(
          `Strategy "${strategy.name}" already exists. Please use a different name.`
        )
      }

      // Add new strategy
      const updatedStrategies = [...strategies, strategy]

      // Build updated config
      const nextConfig = {
        ...projectConfig,
        rag: {
          ...rag,
          data_processing_strategies: updatedStrategies,
        },
      }

      // Update project via API
      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    onSuccess: () => {
      // Invalidate project config to trigger refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error) => {
      console.error('Failed to create strategy:', error)
    },
  })

  /**
   * Update an existing data processing strategy
   */
  const updateStrategy = useMutation({
    mutationFn: async ({
      strategyName,
      updates,
      projectConfig,
    }: {
      strategyName: string
      updates: Partial<DataProcessingStrategy>
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for strategy update')
      }

      // Get or initialize rag.data_processing_strategies
      const rag = projectConfig.rag || {}
      const strategies = rag.data_processing_strategies || []

      // Find strategy by name
      const strategyIndex = strategies.findIndex(
        (s: any) => s.name === strategyName
      )

      if (strategyIndex === -1) {
        throw new Error(`Strategy "${strategyName}" not found in config`)
      }

      // Update the strategy
      const updatedStrategies = [...strategies]
      updatedStrategies[strategyIndex] = {
        ...updatedStrategies[strategyIndex],
        ...updates,
      }

      // Build updated config
      const nextConfig = {
        ...projectConfig,
        rag: {
          ...rag,
          data_processing_strategies: updatedStrategies,
        },
      }

      // Update project via API
      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    onSuccess: () => {
      // Invalidate project config to trigger refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error) => {
      console.error('Failed to update strategy:', error)
    },
  })

  /**
   * Delete a data processing strategy
   */
  const deleteStrategy = useMutation({
    mutationFn: async ({
      strategyName,
      projectConfig,
    }: {
      strategyName: string
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for strategy deletion')
      }

      // Get or initialize rag.data_processing_strategies
      const rag = projectConfig.rag || {}
      const strategies = rag.data_processing_strategies || []

      // Filter out the strategy
      const updatedStrategies = strategies.filter(
        (s: any) => s.name !== strategyName
      )

      if (updatedStrategies.length === strategies.length) {
        throw new Error(`Strategy "${strategyName}" not found in config`)
      }

      // Build updated config
      const nextConfig = {
        ...projectConfig,
        rag: {
          ...rag,
          data_processing_strategies: updatedStrategies,
        },
      }

      // Update project via API
      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    onSuccess: () => {
      // Invalidate project config to trigger refetch
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error) => {
      console.error('Failed to delete strategy:', error)
    },
  })

  return {
    createStrategy,
    updateStrategy,
    deleteStrategy,
    isUpdating:
      createStrategy.isPending ||
      updateStrategy.isPending ||
      deleteStrategy.isPending,
  }
}
