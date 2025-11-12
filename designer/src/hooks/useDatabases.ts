import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../api/client'

/**
 * Database information structure
 */
export interface DatabaseInfo {
  name: string
  type: string
  is_default: boolean
  embedding_strategies: Array<{
    name: string
    type: string
    priority: number
    is_default: boolean
  }>
  retrieval_strategies: Array<{
    name: string
    type: string
    is_default: boolean
  }>
}

/**
 * Response structure for databases API
 */
export interface DatabasesResponse {
  databases: DatabaseInfo[]
  default_database: string | null
}

/**
 * Query keys for database-related queries
 */
export const databaseKeys = {
  all: ['databases'] as const,
  lists: () => [...databaseKeys.all, 'list'] as const,
  list: (namespace: string, projectId: string) =>
    [...databaseKeys.lists(), namespace, projectId] as const,
}

/**
 * Fetch databases for a project
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @returns Promise<DatabasesResponse> - List of databases with default
 */
export async function getDatabases(
  namespace: string,
  projectId: string
): Promise<DatabasesResponse> {
  const response = await apiClient.get<DatabasesResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}/rag/databases`
  )
  return response.data
}

/**
 * Hook to fetch databases for a project
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param enabled - Whether the query should be enabled (default: true)
 * @returns Query result with databases list
 */
export const useDatabases = (
  namespace: string,
  projectId: string,
  enabled = true
) => {
  return useQuery({
    queryKey: databaseKeys.list(namespace, projectId),
    queryFn: () => getDatabases(namespace, projectId),
    enabled: enabled && !!namespace && !!projectId,
    staleTime: 5 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
  })
}

export default {
  useDatabases,
  getDatabases,
  databaseKeys,
}
