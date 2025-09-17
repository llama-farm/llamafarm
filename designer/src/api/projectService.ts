import { apiClient } from './client'
import {
  ListProjectsResponse,
  CreateProjectRequest,
  CreateProjectResponse,
  GetProjectResponse,
  UpdateProjectRequest,
  UpdateProjectResponse,
  DeleteProjectResponse,
} from '../types/project'

/**
 * List all projects in a namespace
 * @param namespace - The namespace to list projects for
 * @returns Promise<ListProjectsResponse> - List of projects with total count
 */
export async function listProjects(namespace: string): Promise<ListProjectsResponse> {
  const response = await apiClient.get<ListProjectsResponse>(`/projects/${namespace}`)
  return response.data
}

/**
 * Create a new project in a namespace
 * @param namespace - The namespace to create the project in
 * @param request - The project creation request
 * @returns Promise<CreateProjectResponse> - The created project
 */
export async function createProject(
  namespace: string, 
  request: CreateProjectRequest
): Promise<CreateProjectResponse> {
  const response = await apiClient.post<CreateProjectResponse>(`/projects/${namespace}`, request)
  return response.data
}

/**
 * Get a single project by namespace and project ID
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @returns Promise<GetProjectResponse> - The requested project
 */
export async function getProject(namespace: string, projectId: string): Promise<GetProjectResponse> {
  const response = await apiClient.get<GetProjectResponse>(`/projects/${namespace}/${encodeURIComponent(projectId)}`)
  return response.data
}

/**
 * Update an existing project
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param request - The project update request
 * @returns Promise<UpdateProjectResponse> - The updated project
 */
export async function updateProject(
  namespace: string, 
  projectId: string, 
  request: UpdateProjectRequest
): Promise<UpdateProjectResponse> {
  const response = await apiClient.put<UpdateProjectResponse>(`/projects/${namespace}/${encodeURIComponent(projectId)}`, request)
  return response.data
}

/**
 * Delete a project
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @returns Promise<DeleteProjectResponse> - The deleted project
 */
export async function deleteProject(namespace: string, projectId: string): Promise<DeleteProjectResponse> {
  const response = await apiClient.delete<DeleteProjectResponse>(`/projects/${namespace}/${encodeURIComponent(projectId)}`)
  return response.data
}

/**
 * Default export with all project service functions
 */
export default {
  listProjects,
  createProject,
  getProject,
  updateProject,
  deleteProject,
}
