import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  createProject,
  getProject,
  listProjects,
  updateProject,
  type CreateProjectRequest,
  type CreateProjectResponse,
  type GetProjectResponse,
  type ListProjectsResponse,
  type UpdateProjectRequest,
  type UpdateProjectResponse,
  type Project,
} from '../api/projects'

// Query key factory
export const projectsKeys = {
  all: ['projects'] as const,
  lists: () => [...projectsKeys.all, 'list'] as const,
  list: (namespace: string) => [...projectsKeys.lists(), { namespace }] as const,
  details: () => [...projectsKeys.all, 'detail'] as const,
  detail: (namespace: string, projectId: string) =>
    [...projectsKeys.details(), { namespace, projectId }] as const,
}

// GET: list projects
export function useProjects(namespace: string | undefined) {
  return useQuery<ListProjectsResponse, Error>({
    queryKey: namespace ? projectsKeys.list(namespace) : projectsKeys.lists(),
    queryFn: () => {
      if (!namespace) throw new Error('namespace is required')
      return listProjects(namespace)
    },
    enabled: Boolean(namespace),
    staleTime: 30_000,
  })
}

// GET: single project
export function useProject(namespace: string | undefined, projectId: string | undefined) {
  return useQuery<GetProjectResponse, Error>({
    queryKey: namespace && projectId ? projectsKeys.detail(namespace, projectId) : projectsKeys.details(),
    queryFn: () => {
      if (!namespace || !projectId) throw new Error('namespace and projectId are required')
      return getProject(namespace, projectId)
    },
    enabled: Boolean(namespace && projectId),
    staleTime: 30_000,
  })
}

// POST: create project
export function useCreateProject() {
  const qc = useQueryClient()
  return useMutation<
    CreateProjectResponse,
    Error,
    { namespace: string; body: CreateProjectRequest },
    { previous?: ListProjectsResponse }
  >(
    {
      mutationFn: ({ namespace, body }) => createProject(namespace, body),
      // Optimistic update: add project to list cache
      onMutate: async ({ namespace, body }) => {
        const listKey = projectsKeys.list(namespace)
        await qc.cancelQueries({ queryKey: listKey })

        const previous = qc.getQueryData<ListProjectsResponse>(listKey)
        const optimisticProject: Project | undefined = body.name
          ? { namespace, name: body.name, config: {} }
          : undefined

        if (previous && optimisticProject) {
          qc.setQueryData<ListProjectsResponse>(listKey, {
            total: previous.total + 1,
            projects: [...previous.projects, optimisticProject],
          })
        }

        return { previous }
      },
      onError: (_err, { namespace }, ctx) => {
        const listKey = projectsKeys.list(namespace)
        if (ctx && ctx.previous) {
          qc.setQueryData(listKey, ctx.previous)
        }
      },
      onSuccess: (data, { namespace }) => {
        // Ensure detail cache populated
        qc.setQueryData(projectsKeys.detail(namespace, data.project.name), { project: data.project })
      },
      onSettled: (_data, _err, { namespace }) => {
        qc.invalidateQueries({ queryKey: projectsKeys.list(namespace) })
      },
    }
  )
}

// PUT: update project (full config)
export function useUpdateProject() {
  const qc = useQueryClient()
  return useMutation<
    UpdateProjectResponse,
    Error,
    { namespace: string; projectId: string; body: UpdateProjectRequest },
    { previous?: GetProjectResponse }
  >({
    mutationFn: ({ namespace, projectId, body }) => updateProject(namespace, projectId, body),
    onMutate: async ({ namespace, projectId, body }) => {
      const detailKey = projectsKeys.detail(namespace, projectId)
      await qc.cancelQueries({ queryKey: detailKey })

      const previous = qc.getQueryData<GetProjectResponse>(detailKey)
      if (previous) {
        qc.setQueryData<GetProjectResponse>(detailKey, {
          project: { ...previous.project, config: body.config },
        })
      }
      return { previous }
    },
    onError: (_err, { namespace, projectId }, ctx) => {
      const detailKey = projectsKeys.detail(namespace, projectId)
      if (ctx && ctx.previous) {
        qc.setQueryData(detailKey, ctx.previous)
      }
    },
    onSuccess: (data, { namespace, projectId }) => {
      // Keep detail in sync
      qc.setQueryData<GetProjectResponse>(projectsKeys.detail(namespace, projectId), {
        project: data.project,
      })
    },
    onSettled: (_data, _err, { namespace, projectId }) => {
      qc.invalidateQueries({ queryKey: projectsKeys.detail(namespace, projectId) })
      qc.invalidateQueries({ queryKey: projectsKeys.list(namespace) })
    },
  })
}


