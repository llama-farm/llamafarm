import axios, { AxiosInstance } from 'axios'

// TODO: Move to .env
// Environment constants
export const API_HOST = 'http://localhost:8000'
export const API_PREFIX = '/v1'
export const API_BASE_URL = `${API_HOST}${API_PREFIX}`

// Axios instance
const http: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
})

// Types aligned with server/api/routers/projects/projects.py
export interface ProjectConfig {
  // Config is a full LlamaFarmConfig object on the server; keep flexible on the client
  [key: string]: unknown
}

export interface Project {
  namespace: string
  name: string
  config: ProjectConfig
}

export interface ListProjectsResponse {
  total: number
  projects: Project[]
}

export interface CreateProjectRequest {
  name: string
  // Server expects a string template name; allow undefined
  config_template?: string
}

export interface CreateProjectResponse {
  project: Project
}

export interface GetProjectResponse {
  project: Project
}

export interface UpdateProjectRequest {
  // Full replacement of the project's configuration
  config: ProjectConfig
}

export interface UpdateProjectResponse {
  project: Project
}

// Types aligned with server/api/routers/projects/projects.py


// Layer 1: Plain API functions
export async function listProjects(namespace: string): Promise<ListProjectsResponse> {
  const res = await http.get<ListProjectsResponse>(`/projects/${encodeURIComponent(namespace)}`)
  return res.data
}

export async function createProject(
  namespace: string,
  body: CreateProjectRequest
): Promise<CreateProjectResponse> {
  const res = await http.post<CreateProjectResponse>(
    `/projects/${encodeURIComponent(namespace)}`,
    body
  )
  return res.data
}

export async function getProject(
  namespace: string,
  projectId: string
): Promise<GetProjectResponse> {
  const res = await http.get<GetProjectResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}`
  )
  return res.data
}

export async function updateProject(
  namespace: string,
  projectId: string,
  body: UpdateProjectRequest
): Promise<UpdateProjectResponse> {
  const res = await http.put<UpdateProjectResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}`,
    body
  )
  return res.data
}

export default {
  listProjects,
  createProject,
  getProject,
  updateProject,
}


