import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useUpdateProject, projectKeys } from './useProjects'

/**
 * Type for a database
 */
export type Database = {
  name: string
  type: 'ChromaStore' | 'QdrantStore'
  config?: Record<string, any>
  default_embedding_strategy?: string
  default_retrieval_strategy?: string
  embedding_strategies?: Array<{
    name: string
    type: string
    priority?: number
    config?: Record<string, any>
  }>
  retrieval_strategies?: Array<{
    name: string
    type: string
    config?: Record<string, any>
    default?: boolean
  }>
}

/**
 * Hook to manage databases in the config.
 *
 * NOTE: Uses client-side read-modify-write pattern which may have race conditions
 * with concurrent modifications. Consider backend atomic endpoints for production.
 */
export const useDatabaseManager = (namespace: string, projectId: string) => {
  const queryClient = useQueryClient()
  const updateProjectMutation = useUpdateProject()

  const baseMutationOptions = {
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: projectKeys.detail(namespace, projectId),
      })
    },
    onError: (error: Error) => {
      console.error('Database operation failed:', error)
    },
  }

  const createDatabase = useMutation({
    mutationFn: async ({
      database,
      projectConfig,
    }: {
      database: Database
      projectConfig: any
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for database creation')
      }

      const databases = projectConfig.rag?.databases || []

      if (databases.some((db: any) => db.name === database.name)) {
        throw new Error(
          `Database "${database.name}" already exists. Please use a different name.`
        )
      }

      const nextConfig = {
        ...projectConfig,
        rag: {
          ...projectConfig.rag,
          databases: [...databases, database],
        },
      }

      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    ...baseMutationOptions,
  })

  const updateDatabase = useMutation({
    mutationFn: async ({
      oldName,
      updates,
      projectConfig,
      datasetUpdates,
    }: {
      oldName: string
      updates: Partial<Database>
      projectConfig: any
      datasetUpdates?: Array<{ name: string; database: string }>
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for database update')
      }

      const databases = projectConfig.rag?.databases || []
      const databaseIndex = databases.findIndex(
        (db: any) => db.name === oldName
      )

      if (databaseIndex === -1) {
        throw new Error(`Database "${oldName}" not found in config`)
      }

      if (updates.name && updates.name !== oldName) {
        if (databases.some((db: any) => db.name === updates.name)) {
          throw new Error(
            `Database "${updates.name}" already exists. Please use a different name.`
          )
        }
      }

      const updatedDatabases = [...databases]
      updatedDatabases[databaseIndex] = {
        ...updatedDatabases[databaseIndex],
        ...updates,
      }

      let updatedDatasets = projectConfig.datasets || []
      if (datasetUpdates && datasetUpdates.length > 0) {
        updatedDatasets = updatedDatasets.map((ds: any) => {
          const update = datasetUpdates.find((u) => u.name === ds.name)
          return update ? { ...ds, database: update.database } : ds
        })
      }

      const nextConfig = {
        ...projectConfig,
        rag: { ...projectConfig.rag, databases: updatedDatabases },
        datasets: updatedDatasets,
      }

      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    ...baseMutationOptions,
  })

  const deleteDatabase = useMutation({
    mutationFn: async ({
      databaseName,
      projectConfig,
      reassignTo,
    }: {
      databaseName: string
      projectConfig: any
      reassignTo?: string
    }) => {
      if (!namespace || !projectId || !projectConfig) {
        throw new Error('Missing required parameters for database deletion')
      }

      const databases = projectConfig.rag?.databases || []
      const updatedDatabases = databases.filter(
        (db: any) => db.name !== databaseName
      )

      if (updatedDatabases.length === databases.length) {
        throw new Error(`Database "${databaseName}" not found in config`)
      }

      let updatedDatasets = projectConfig.datasets || []
      if (reassignTo) {
        updatedDatasets = updatedDatasets.map((ds: any) =>
          ds.database === databaseName ? { ...ds, database: reassignTo } : ds
        )
      }

      const nextConfig = {
        ...projectConfig,
        rag: { ...projectConfig.rag, databases: updatedDatabases },
        datasets: updatedDatasets,
      }

      return await updateProjectMutation.mutateAsync({
        namespace,
        projectId,
        request: { config: nextConfig },
      })
    },
    ...baseMutationOptions,
  })

  return {
    createDatabase,
    updateDatabase,
    deleteDatabase,
    isUpdating:
      createDatabase.isPending ||
      updateDatabase.isPending ||
      deleteDatabase.isPending,
  }
}
