import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import projectService from '../api/projectService'
import {
  CreateProjectRequest,
  UpdateProjectRequest,
} from '../types/project'

/**
 * Query keys for project-related queries
 * Follows the pattern used in existing chat hooks
 */
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (namespace: string) => [...projectKeys.lists(), namespace] as const,
  details: () => [...projectKeys.all, 'detail'] as const,
  detail: (namespace: string, projectId: string) => [...projectKeys.details(), namespace, projectId] as const,
}

/**
 * Hook to fetch all projects in a namespace
 * @param namespace - The namespace to fetch projects for
 * @returns Query result with projects list
 */
export const useProjects = (namespace: string) => {
  return useQuery({
    queryKey: projectKeys.list(namespace),
    queryFn: () => projectService.listProjects(namespace),
    enabled: !!namespace, // Only run query if namespace is provided
  })
}

/**
 * Hook to fetch a single project
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param enabled - Whether the query should be enabled (default: true)
 * @returns Query result with project details
 */
export const useProject = (namespace: string, projectId: string, enabled = true) => {
  return useQuery({
    queryKey: projectKeys.detail(namespace, projectId),
    queryFn: () => projectService.getProject(namespace, projectId),
    enabled: enabled && !!namespace && !!projectId,
  })
}

/**
 * Hook to create a new project
 * @returns Mutation function for creating projects
 */
export const useCreateProject = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ namespace, request }: { namespace: string; request: CreateProjectRequest }) => 
      projectService.createProject(namespace, request),
    onSuccess: (data, variables) => {
      // Invalidate and refetch projects list for the namespace
      queryClient.invalidateQueries({ queryKey: projectKeys.list(variables.namespace) })
      
      // Optionally add the new project to the cache
      queryClient.setQueryData(
        projectKeys.detail(variables.namespace, data.project.name),
        { project: data.project }
      )
    },
    onError: (error) => {
      // Error handling is already done by apiClient interceptor
      console.error('Failed to create project:', error)
    }
  })
}

/**
 * Hook to update an existing project
 * @returns Mutation function for updating projects
 */
export const useUpdateProject = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ 
      namespace, 
      projectId, 
      request 
    }: { 
      namespace: string; 
      projectId: string; 
      request: UpdateProjectRequest 
    }) => projectService.updateProject(namespace, projectId, request),
    onSuccess: (data, variables) => {
      // Update the specific project in cache
      queryClient.setQueryData(
        projectKeys.detail(variables.namespace, variables.projectId),
        { project: data.project }
      )
      
      // Invalidate projects list to ensure consistency
      queryClient.invalidateQueries({ queryKey: projectKeys.list(variables.namespace) })
    },
    onError: (error) => {
      console.error('Failed to update project:', error)
    }
  })
}

/**
 * Hook to delete a project
 * @returns Mutation function for deleting projects
 */
export const useDeleteProject = () => {
  const queryClient = useQueryClient()
  
  return useMutation({
    mutationFn: ({ namespace, projectId }: { namespace: string; projectId: string }) => 
      projectService.deleteProject(namespace, projectId),
    onSuccess: (_, variables) => {
      // Remove the deleted project from cache
      queryClient.removeQueries({ 
        queryKey: projectKeys.detail(variables.namespace, variables.projectId) 
      })
      
      // Invalidate projects list
      queryClient.invalidateQueries({ queryKey: projectKeys.list(variables.namespace) })
    },
    onError: (error) => {
      console.error('Failed to delete project:', error)
    }
  })
}

/**
 * Hook to get project mutation loading states
 * Useful for components that need to show loading states
 */
export const useProjectMutations = () => {
  const createMutation = useCreateProject()
  const updateMutation = useUpdateProject()
  const deleteMutation = useDeleteProject()
  
  return {
    create: createMutation,
    update: updateMutation,
    delete: deleteMutation,
    isLoading: createMutation.isPending || updateMutation.isPending || deleteMutation.isPending,
    error: createMutation.error || updateMutation.error || deleteMutation.error,
  }
}
