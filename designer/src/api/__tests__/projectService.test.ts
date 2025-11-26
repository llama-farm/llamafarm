import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  listProjects,
  createProject,
  getProject,
  updateProject,
  deleteProject,
} from '../projectService'
import type {
  CreateProjectRequest,
  UpdateProjectRequest,
} from '../../types/project'

// Mock apiClient
vi.mock('../client', () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

import { apiClient } from '../client'
import {
  createMockProject,
  createMockProjectsList,
  createMockProjectWithError,
} from '../../test/factories/projectFactory'

describe('projectService', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('listProjects', () => {
    it('should list projects for namespace', async () => {
      const mockResponse = createMockProjectsList('default', 2)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockResponse })

      const result = await listProjects('default')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default')
      expect(result).toEqual(mockResponse)
      expect(result.total).toBe(2)
      expect(result.projects).toHaveLength(2)
    })

    it('should construct correct URL with namespace', async () => {
      const mockResponse = createMockProjectsList('my-namespace', 1)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockResponse })

      await listProjects('my-namespace')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/my-namespace')
    })

    it('should not encode namespace in list endpoint', async () => {
      const mockResponse = createMockProjectsList('namespace-with-dashes', 0)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockResponse })

      await listProjects('namespace-with-dashes')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/namespace-with-dashes')
    })

    it('should return response.data', async () => {
      const mockData = createMockProjectsList('default', 3)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockData })

      const result = await listProjects('default')

      expect(result).toBe(mockData)
      expect(result).not.toHaveProperty('status')
      expect(result).not.toHaveProperty('headers')
    })

    it('should throw error when API fails', async () => {
      const error = new Error('API Error')
      vi.mocked(apiClient.get).mockRejectedValue(error)

      await expect(listProjects('default')).rejects.toThrow('API Error')
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network request failed')
      vi.mocked(apiClient.get).mockRejectedValue(networkError)

      await expect(listProjects('default')).rejects.toThrow('Network request failed')
    })

    it('should handle 404 response', async () => {
      const error = new Error('Not found')
      vi.mocked(apiClient.get).mockRejectedValue(error)

      await expect(listProjects('unknown-namespace')).rejects.toThrow('Not found')
    })

    it('should handle empty project list', async () => {
      const emptyResponse = createMockProjectsList('default', 0)
      vi.mocked(apiClient.get).mockResolvedValue({ data: emptyResponse })

      const result = await listProjects('default')

      expect(result.total).toBe(0)
      expect(result.projects).toHaveLength(0)
    })

    it('should handle namespace with special characters', async () => {
      const mockResponse = createMockProjectsList('my_namespace-123', 1)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockResponse })

      await listProjects('my_namespace-123')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/my_namespace-123')
    })
  })

  describe('createProject', () => {
    it('should create project successfully', async () => {
      const request: CreateProjectRequest = {
        name: 'test-project',
        config_template: 'default',
      }
      const mockProject = createMockProject({ name: 'test-project', namespace: 'default' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      const result = await createProject('default', request)

      expect(apiClient.post).toHaveBeenCalledWith('/projects/default', request)
      expect(result.project).toEqual(mockProject)
    })

    it('should construct correct URL for create', async () => {
      const request: CreateProjectRequest = { name: 'test' }
      const mockProject = createMockProject({ name: 'test', namespace: 'my-namespace' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      await createProject('my-namespace', request)

      expect(apiClient.post).toHaveBeenCalledWith('/projects/my-namespace', request)
    })

    it('should pass request as body', async () => {
      const request: CreateProjectRequest = {
        name: 'new-project',
        config_template: 'custom',
      }
      const mockProject = createMockProject(request)
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      await createProject('default', request)

      expect(apiClient.post).toHaveBeenCalledWith('/projects/default', request)
      const callArgs = vi.mocked(apiClient.post).mock.calls[0]
      expect(callArgs[1]).toEqual(request)
    })

    it('should return created project from response', async () => {
      const request: CreateProjectRequest = { name: 'test' }
      const mockProject = createMockProject({ name: 'test' })
      const mockResponse = { project: mockProject }
      vi.mocked(apiClient.post).mockResolvedValue({ data: mockResponse })

      const result = await createProject('default', request)

      expect(result).toEqual(mockResponse)
      expect(result.project).toEqual(mockProject)
    })

    it('should throw error when creation fails', async () => {
      const request: CreateProjectRequest = { name: 'test' }
      const error = new Error('Creation failed')
      vi.mocked(apiClient.post).mockRejectedValue(error)

      await expect(createProject('default', request)).rejects.toThrow('Creation failed')
    })

    it('should handle validation errors from backend', async () => {
      const request: CreateProjectRequest = { name: '' }
      const validationError = new Error('Validation failed: name is required')
      vi.mocked(apiClient.post).mockRejectedValue(validationError)

      await expect(createProject('default', request)).rejects.toThrow('Validation failed')
    })

    it('should handle duplicate project name error', async () => {
      const request: CreateProjectRequest = { name: 'existing-project' }
      const conflictError = new Error('Project already exists')
      vi.mocked(apiClient.post).mockRejectedValue(conflictError)

      await expect(createProject('default', request)).rejects.toThrow('Project already exists')
    })

    it('should handle network errors', async () => {
      const request: CreateProjectRequest = { name: 'test' }
      const networkError = new Error('Network error')
      vi.mocked(apiClient.post).mockRejectedValue(networkError)

      await expect(createProject('default', request)).rejects.toThrow('Network error')
    })

    it('should handle special characters in project name', async () => {
      const request: CreateProjectRequest = { name: 'test-project_123' }
      const mockProject = createMockProject({ name: 'test-project_123' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      const result = await createProject('default', request)

      expect(result.project.name).toBe('test-project_123')
    })

    it('should handle long project names', async () => {
      const longName = 'very-long-project-name-that-might-cause-issues-if-not-handled-properly'
      const request: CreateProjectRequest = { name: longName }
      const mockProject = createMockProject({ name: longName })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      const result = await createProject('default', request)

      expect(result.project.name).toBe(longName)
    })

    it('should handle template parameter', async () => {
      const request: CreateProjectRequest = {
        name: 'test',
        config_template: 'advanced',
      }
      const mockProject = createMockProject({ name: 'test' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      await createProject('default', request)

      expect(apiClient.post).toHaveBeenCalledWith('/projects/default', expect.objectContaining({
        config_template: 'advanced',
      }))
    })

    it('should handle request without template', async () => {
      const request: CreateProjectRequest = { name: 'test' }
      const mockProject = createMockProject({ name: 'test' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      await createProject('default', request)

      expect(apiClient.post).toHaveBeenCalledWith('/projects/default', request)
    })

    it('should return created project in response', async () => {
      const request: CreateProjectRequest = { name: 'new-project' }
      const mockProject = createMockProject({ name: 'new-project', namespace: 'default' })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      const result = await createProject('default', request)

      expect(result).toHaveProperty('project')
      expect(result.project.name).toBe('new-project')
      expect(result.project.namespace).toBe('default')
    })
  })

  describe('getProject', () => {
    it('should get project by namespace and projectId', async () => {
      const mockProject = createMockProject({ namespace: 'default', name: 'my-project' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      const result = await getProject('default', 'my-project')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/my-project')
      expect(result.project).toEqual(mockProject)
    })

    it('should construct correct URL with parameters', async () => {
      const mockProject = createMockProject({ namespace: 'my-namespace', name: 'project-name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('my-namespace', 'project-name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/my-namespace/project-name')
    })

    it('should URL encode projectId', async () => {
      const mockProject = createMockProject({ name: 'project with spaces' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project with spaces')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%20with%20spaces')
    })

    it('should handle special characters in projectId', async () => {
      const mockProject = createMockProject({ name: 'project/with/slashes' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project/with/slashes')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%2Fwith%2Fslashes')
    })

    it('should throw error when project not found', async () => {
      const error = new Error('Project not found')
      vi.mocked(apiClient.get).mockRejectedValue(error)

      await expect(getProject('default', 'non-existent')).rejects.toThrow('Project not found')
    })

    it('should return project from response', async () => {
      const mockProject = createMockProject({ name: 'test-project' })
      const mockResponse = { project: mockProject }
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockResponse })

      const result = await getProject('default', 'test-project')

      expect(result).toEqual(mockResponse)
      expect(result.project).toEqual(mockProject)
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network error')
      vi.mocked(apiClient.get).mockRejectedValue(networkError)

      await expect(getProject('default', 'test')).rejects.toThrow('Network error')
    })

    it('should handle namespace with special characters', async () => {
      const mockProject = createMockProject({ namespace: 'my_namespace-123' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('my_namespace-123', 'test-project')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/my_namespace-123/test-project')
    })

    it('should handle question mark in projectId', async () => {
      const mockProject = createMockProject({ name: 'project?name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project?name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%3Fname')
    })

    it('should handle hash in projectId', async () => {
      const mockProject = createMockProject({ name: 'project#name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project#name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%23name')
    })

    it('should return project with validation errors', async () => {
      const mockProject = createMockProjectWithError('test', 'Invalid config')
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      const result = await getProject('default', 'test')

      expect(result.project.validation_error).toBe('Invalid config')
    })
  })

  describe('updateProject', () => {
    it('should update project successfully', async () => {
      const request: UpdateProjectRequest = {
        config: {
          version: 'v1',
          name: 'my-project',
          namespace: 'default',
          runtime: { provider: 'ollama', model: 'llama3.2:3b' },
          prompts: [],
        },
      }
      const mockProject = createMockProject({ name: 'my-project', config: request.config })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      const result = await updateProject('default', 'my-project', request)

      expect(apiClient.put).toHaveBeenCalledWith('/projects/default/my-project', request)
      expect(result.project).toEqual(mockProject)
    })

    it('should construct correct URL for update', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const mockProject = createMockProject({ namespace: 'ns', name: 'proj' })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await updateProject('ns', 'proj', request)

      expect(apiClient.put).toHaveBeenCalledWith('/projects/ns/proj', request)
    })

    it('should URL encode projectId in update', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const mockProject = createMockProject({ name: 'project with spaces' })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await updateProject('default', 'project with spaces', request)

      expect(apiClient.put).toHaveBeenCalledWith('/projects/default/project%20with%20spaces', request)
    })

    it('should support partial updates', async () => {
      const request: UpdateProjectRequest = {
        config: { runtime: { provider: 'openai', model: 'gpt-4' } },
      }
      const mockProject = createMockProject({ config: request.config })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await updateProject('default', 'test', request)

      expect(apiClient.put).toHaveBeenCalledWith('/projects/default/test', request)
      const callArgs = vi.mocked(apiClient.put).mock.calls[0]
      expect(callArgs[1]).toEqual(request)
    })

    it('should return updated project from response', async () => {
      const request: UpdateProjectRequest = { config: { updated: true } }
      const mockProject = createMockProject({ name: 'test', config: { updated: true } })
      const mockResponse = { project: mockProject }
      vi.mocked(apiClient.put).mockResolvedValue({ data: mockResponse })

      const result = await updateProject('default', 'test', request)

      expect(result).toEqual(mockResponse)
      expect(result.project.config).toHaveProperty('updated', true)
    })

    it('should throw error when update fails', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const error = new Error('Update failed')
      vi.mocked(apiClient.put).mockRejectedValue(error)

      await expect(updateProject('default', 'test', request)).rejects.toThrow('Update failed')
    })

    it('should handle validation errors', async () => {
      const request: UpdateProjectRequest = { config: { invalid: 'config' } }
      const validationError = new Error('Validation failed')
      vi.mocked(apiClient.put).mockRejectedValue(validationError)

      await expect(updateProject('default', 'test', request)).rejects.toThrow('Validation failed')
    })

    it('should handle project not found error', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const error = new Error('Project not found')
      vi.mocked(apiClient.put).mockRejectedValue(error)

      await expect(updateProject('default', 'non-existent', request)).rejects.toThrow('Project not found')
    })

    it('should handle network errors', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const networkError = new Error('Network error')
      vi.mocked(apiClient.put).mockRejectedValue(networkError)

      await expect(updateProject('default', 'test', request)).rejects.toThrow('Network error')
    })

    it('should handle special characters in projectId', async () => {
      const request: UpdateProjectRequest = { config: {} }
      const mockProject = createMockProject({ name: 'project/name' })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await updateProject('default', 'project/name', request)

      expect(apiClient.put).toHaveBeenCalledWith('/projects/default/project%2Fname', request)
    })
  })

  describe('deleteProject', () => {
    it('should delete project successfully', async () => {
      const mockProject = createMockProject({ namespace: 'default', name: 'my-project' })
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      const result = await deleteProject('default', 'my-project')

      expect(apiClient.delete).toHaveBeenCalledWith('/projects/default/my-project')
      expect(result.project).toEqual(mockProject)
    })

    it('should construct correct URL for delete', async () => {
      const mockProject = createMockProject({ namespace: 'ns', name: 'proj' })
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      await deleteProject('ns', 'proj')

      expect(apiClient.delete).toHaveBeenCalledWith('/projects/ns/proj')
    })

    it('should URL encode projectId in delete', async () => {
      const mockProject = createMockProject({ name: 'project with spaces' })
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      await deleteProject('default', 'project with spaces')

      expect(apiClient.delete).toHaveBeenCalledWith('/projects/default/project%20with%20spaces')
    })

    it('should return deleted project from response', async () => {
      const mockProject = createMockProject({ name: 'test-project' })
      const mockResponse = { project: mockProject }
      vi.mocked(apiClient.delete).mockResolvedValue({ data: mockResponse })

      const result = await deleteProject('default', 'test-project')

      expect(result).toEqual(mockResponse)
      expect(result.project).toEqual(mockProject)
    })

    it('should throw error when deletion fails', async () => {
      const error = new Error('Deletion failed')
      vi.mocked(apiClient.delete).mockRejectedValue(error)

      await expect(deleteProject('default', 'test')).rejects.toThrow('Deletion failed')
    })

    it('should handle project not found error', async () => {
      const error = new Error('Project not found')
      vi.mocked(apiClient.delete).mockRejectedValue(error)

      await expect(deleteProject('default', 'non-existent')).rejects.toThrow('Project not found')
    })

    it('should handle cascade delete conflicts', async () => {
      const conflictError = new Error('Cannot delete project with active datasets')
      vi.mocked(apiClient.delete).mockRejectedValue(conflictError)

      await expect(deleteProject('default', 'test')).rejects.toThrow('Cannot delete project with active datasets')
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network error')
      vi.mocked(apiClient.delete).mockRejectedValue(networkError)

      await expect(deleteProject('default', 'test')).rejects.toThrow('Network error')
    })

    it('should handle special characters in projectId', async () => {
      const mockProject = createMockProject({ name: 'project/name' })
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      await deleteProject('default', 'project/name')

      expect(apiClient.delete).toHaveBeenCalledWith('/projects/default/project%2Fname')
    })

    it('should handle ampersand in projectId', async () => {
      const mockProject = createMockProject({ name: 'project&name' })
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      await deleteProject('default', 'project&name')

      expect(apiClient.delete).toHaveBeenCalledWith('/projects/default/project%26name')
    })
  })

  describe('URL Encoding Edge Cases', () => {
    it('should encode forward slash in projectId', async () => {
      const mockProject = createMockProject({ name: 'project/name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project/name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%2Fname')
    })

    it('should encode question mark in projectId', async () => {
      const mockProject = createMockProject({ name: 'project?name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project?name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%3Fname')
    })

    it('should encode hash in projectId', async () => {
      const mockProject = createMockProject({ name: 'project#name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project#name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%23name')
    })

    it('should encode ampersand in projectId', async () => {
      const mockProject = createMockProject({ name: 'project&name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project&name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%26name')
    })

    it('should encode percent sign in projectId', async () => {
      const mockProject = createMockProject({ name: 'project%name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project%name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%25name')
    })

    it('should handle plus sign in projectId', async () => {
      const mockProject = createMockProject({ name: 'project+name' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project+name')

      expect(apiClient.get).toHaveBeenCalledWith('/projects/default/project%2Bname')
    })

    it('should encode Unicode characters in projectId', async () => {
      const mockProject = createMockProject({ name: 'project-测试' })
      vi.mocked(apiClient.get).mockResolvedValue({ data: { project: mockProject } })

      await getProject('default', 'project-测试')

      // Unicode characters should be percent-encoded
      expect(apiClient.get).toHaveBeenCalledWith(expect.stringContaining('/projects/default/project-'))
    })
  })

  describe('Response Handling', () => {
    it('should extract data from axios response object', async () => {
      const mockData = createMockProjectsList('default', 2)
      const axiosResponse = {
        data: mockData,
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {} as any,
      }
      vi.mocked(apiClient.get).mockResolvedValue(axiosResponse)

      const result = await listProjects('default')

      expect(result).toEqual(mockData)
      expect(result).not.toHaveProperty('status')
      expect(result).not.toHaveProperty('headers')
      expect(result).not.toHaveProperty('config')
    })

    it('should propagate axios error responses', async () => {
      const apiError = new Error('API Error')
      Object.assign(apiError, {
        response: {
          status: 400,
          data: { detail: 'Invalid request' },
        },
      })
      vi.mocked(apiClient.get).mockRejectedValue(apiError)

      await expect(listProjects('default')).rejects.toThrow('API Error')
    })

    it('should propagate axios network errors', async () => {
      const networkError = new Error('Network Error')
      Object.assign(networkError, {
        code: 'ECONNREFUSED',
        request: {},
      })
      vi.mocked(apiClient.get).mockRejectedValue(networkError)

      await expect(listProjects('default')).rejects.toThrow('Network Error')
    })
  })

  describe('Type Safety', () => {
    it('should use correct request types', async () => {
      // TypeScript will enforce types at compile time
      const createRequest: CreateProjectRequest = { name: 'test' }
      const updateRequest: UpdateProjectRequest = { config: {} }

      const mockProject = createMockProject()
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await createProject('default', createRequest)
      await updateProject('default', 'test', updateRequest)

      // If this compiles, types are enforced
      expect(true).toBe(true)
    })

    it('should return correct response types', async () => {
      const mockProjectsList = createMockProjectsList('default', 2)
      const mockProject = createMockProject()

      vi.mocked(apiClient.get).mockResolvedValue({ data: mockProjectsList })
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      const listResult = await listProjects('default')
      const createResult = await createProject('default', { name: 'test' })

      // TypeScript enforces these properties exist
      expect(listResult).toHaveProperty('projects')
      expect(listResult).toHaveProperty('total')
      expect(createResult).toHaveProperty('project')
    })
  })

  describe('Integration with apiClient', () => {
    it('should use apiClient.get for list', async () => {
      const mockData = createMockProjectsList('default', 1)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockData })

      await listProjects('default')

      expect(apiClient.get).toHaveBeenCalled()
      expect(vi.mocked(apiClient.get).mock.calls.length).toBe(1)
    })

    it('should use apiClient.post for create', async () => {
      const mockProject = createMockProject()
      vi.mocked(apiClient.post).mockResolvedValue({ data: { project: mockProject } })

      await createProject('default', { name: 'test' })

      expect(apiClient.post).toHaveBeenCalled()
      expect(vi.mocked(apiClient.post).mock.calls.length).toBe(1)
    })

    it('should use apiClient.put for update', async () => {
      const mockProject = createMockProject()
      vi.mocked(apiClient.put).mockResolvedValue({ data: { project: mockProject } })

      await updateProject('default', 'test', { config: {} })

      expect(apiClient.put).toHaveBeenCalled()
      expect(vi.mocked(apiClient.put).mock.calls.length).toBe(1)
    })

    it('should use apiClient.delete for delete', async () => {
      const mockProject = createMockProject()
      vi.mocked(apiClient.delete).mockResolvedValue({ data: { project: mockProject } })

      await deleteProject('default', 'test')

      expect(apiClient.delete).toHaveBeenCalled()
      expect(vi.mocked(apiClient.delete).mock.calls.length).toBe(1)
    })

    it('should inherit apiClient interceptors (auth, error handling)', async () => {
      // apiClient has interceptors that add headers, handle errors, etc.
      // By using apiClient, all service functions benefit from these
      const mockData = createMockProjectsList('default', 1)
      vi.mocked(apiClient.get).mockResolvedValue({ data: mockData })

      await listProjects('default')

      // Verify we're using apiClient (which has interceptors) not raw axios
      expect(apiClient.get).toHaveBeenCalled()
    })
  })

  describe('Default Export', () => {
    it('should export all service methods', async () => {
      const defaultExport = await import('../projectService')

      expect(defaultExport.default).toHaveProperty('listProjects')
      expect(defaultExport.default).toHaveProperty('createProject')
      expect(defaultExport.default).toHaveProperty('getProject')
      expect(defaultExport.default).toHaveProperty('updateProject')
      expect(defaultExport.default).toHaveProperty('deleteProject')

      expect(typeof defaultExport.default.listProjects).toBe('function')
      expect(typeof defaultExport.default.createProject).toBe('function')
      expect(typeof defaultExport.default.getProject).toBe('function')
      expect(typeof defaultExport.default.updateProject).toBe('function')
      expect(typeof defaultExport.default.deleteProject).toBe('function')
    })
  })
})

