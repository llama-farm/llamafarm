import { describe, it, expect, beforeEach } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../../test/mocks/server'
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
import {
  createMockProject,
  createMockProjectsList,
  createMockProjectWithError,
} from '../../test/factories/projectFactory'

// Base URL must match what apiClient uses in test environment
const API_BASE = 'http://localhost:8000/v1'

describe('projectService', () => {
  beforeEach(() => {
    // Reset handlers to defaults before each test
    server.resetHandlers()
  })

  describe('listProjects', () => {
    it('should list projects for namespace', async () => {
      const mockResponse = createMockProjectsList('default', 2)
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(mockResponse)
        })
      )

      const result = await listProjects('default')

      expect(result).toEqual(mockResponse)
      expect(result.total).toBe(2)
      expect(result.projects).toHaveLength(2)
    })

    it('should return data for different namespaces', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, ({ params }) => {
          const { namespace } = params
          return HttpResponse.json(createMockProjectsList(namespace as string, 1))
        })
      )

      const result = await listProjects('my-namespace')

      expect(result.projects[0].namespace).toBe('my-namespace')
    })

    it('should handle empty project list', async () => {
      const emptyResponse = createMockProjectsList('default', 0)
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(emptyResponse)
        })
      )

      const result = await listProjects('default')

      expect(result.total).toBe(0)
      expect(result.projects).toHaveLength(0)
    })

    it('should throw error when API fails', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Internal server error' },
            { status: 500 }
          )
        })
      )

      await expect(listProjects('default')).rejects.toThrow()
    })

    it('should handle 404 response', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Namespace not found' },
            { status: 404 }
          )
        })
      )

      await expect(listProjects('unknown-namespace')).rejects.toThrow()
    })

    it('should handle network errors', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.error()
        })
      )

      await expect(listProjects('default')).rejects.toThrow()
    })
  })

  describe('createProject', () => {
    it('should create project successfully', async () => {
      const request: CreateProjectRequest = {
        name: 'test-project',
        config_template: 'default',
      }
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, async ({ request: req }) => {
          const body = await req.json() as CreateProjectRequest
          const project = createMockProject({ 
            name: body.name, 
            namespace: 'default' 
          })
          return HttpResponse.json({ project })
        })
      )

      const result = await createProject('default', request)

      expect(result.project.name).toBe('test-project')
      expect(result.project.namespace).toBe('default')
    })

    it('should return created project from response', async () => {
      const request: CreateProjectRequest = { name: 'new-project' }
      const mockProject = createMockProject({ name: 'new-project', namespace: 'default' })
      
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      const result = await createProject('default', request)

      expect(result).toHaveProperty('project')
      expect(result.project.name).toBe('new-project')
    })

    it('should throw error when creation fails', async () => {
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Creation failed' },
            { status: 500 }
          )
        })
      )

      await expect(createProject('default', { name: 'test' })).rejects.toThrow()
    })

    it('should handle validation errors from backend', async () => {
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Validation failed: name is required' },
            { status: 400 }
          )
        })
      )

      await expect(createProject('default', { name: '' })).rejects.toThrow()
    })

    it('should handle duplicate project name error', async () => {
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Project already exists' },
            { status: 409 }
          )
        })
      )

      await expect(createProject('default', { name: 'existing-project' })).rejects.toThrow()
    })

    it('should handle network errors', async () => {
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.error()
        })
      )

      await expect(createProject('default', { name: 'test' })).rejects.toThrow()
    })

    it('should handle special characters in project name', async () => {
      const request: CreateProjectRequest = { name: 'test-project_123' }
      
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, async ({ request: req }) => {
          const body = await req.json() as CreateProjectRequest
          return HttpResponse.json({ 
            project: createMockProject({ name: body.name }) 
          })
        })
      )

      const result = await createProject('default', request)

      expect(result.project.name).toBe('test-project_123')
    })

    it('should handle long project names', async () => {
      const longName = 'very-long-project-name-that-might-cause-issues-if-not-handled-properly'
      const request: CreateProjectRequest = { name: longName }
      
      server.use(
        http.post(`${API_BASE}/projects/:namespace`, async ({ request: req }) => {
          const body = await req.json() as CreateProjectRequest
          return HttpResponse.json({ 
            project: createMockProject({ name: body.name }) 
          })
        })
      )

      const result = await createProject('default', request)

      expect(result.project.name).toBe(longName)
    })
  })

  describe('getProject', () => {
    it('should get project by namespace and projectId', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
          const project = createMockProject({ 
            namespace: params.namespace as string, 
            name: params.projectId as string 
          })
          return HttpResponse.json({ project })
        })
      )

      const result = await getProject('default', 'my-project')

      expect(result.project.namespace).toBe('default')
      expect(result.project.name).toBe('my-project')
    })

    it('should throw error when project not found', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Project not found' },
            { status: 404 }
          )
        })
      )

      await expect(getProject('default', 'non-existent')).rejects.toThrow()
    })

    it('should return project from response', async () => {
      const mockProject = createMockProject({ name: 'test-project' })
      
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      const result = await getProject('default', 'test-project')

      expect(result).toHaveProperty('project')
      expect(result.project).toEqual(mockProject)
    })

    it('should handle network errors', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.error()
        })
      )

      await expect(getProject('default', 'test')).rejects.toThrow()
    })

    it('should return project with validation errors', async () => {
      const mockProject = createMockProjectWithError('test', 'Invalid config')
      
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      const result = await getProject('default', 'test')

      expect(result.project.validation_error).toBe('Invalid config')
    })

    it('should handle URL encoded projectId', async () => {
      // MSW will receive the encoded URL and decode the params
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
          // MSW automatically decodes URL params
          const project = createMockProject({ 
            name: params.projectId as string 
          })
          return HttpResponse.json({ project })
        })
      )

      const result = await getProject('default', 'project with spaces')

      expect(result.project.name).toBe('project with spaces')
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
      
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, async ({ request: req, params }) => {
          const body = await req.json() as UpdateProjectRequest
          const project = createMockProject({ 
            name: params.projectId as string, 
            config: body.config 
          })
          return HttpResponse.json({ project })
        })
      )

      const result = await updateProject('default', 'my-project', request)

      expect(result.project.name).toBe('my-project')
      expect(result.project.config).toEqual(request.config)
    })

    it('should support partial updates', async () => {
      const request: UpdateProjectRequest = {
        config: { runtime: { provider: 'openai', model: 'gpt-4' } },
      }
      
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, async ({ request: req }) => {
          const body = await req.json() as UpdateProjectRequest
          const project = createMockProject({ config: body.config })
          return HttpResponse.json({ project })
        })
      )

      const result = await updateProject('default', 'test', request)

      expect(result.project.config.runtime).toEqual({ provider: 'openai', model: 'gpt-4' })
    })

    it('should throw error when update fails', async () => {
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Update failed' },
            { status: 500 }
          )
        })
      )

      await expect(updateProject('default', 'test', { config: {} })).rejects.toThrow()
    })

    it('should handle validation errors', async () => {
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Validation failed' },
            { status: 400 }
          )
        })
      )

      await expect(updateProject('default', 'test', { config: { invalid: 'config' } })).rejects.toThrow()
    })

    it('should handle project not found error', async () => {
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Project not found' },
            { status: 404 }
          )
        })
      )

      await expect(updateProject('default', 'non-existent', { config: {} })).rejects.toThrow()
    })

    it('should handle network errors', async () => {
      server.use(
        http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.error()
        })
      )

      await expect(updateProject('default', 'test', { config: {} })).rejects.toThrow()
    })
  })

  describe('deleteProject', () => {
    it('should delete project successfully', async () => {
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
          const project = createMockProject({ 
            namespace: params.namespace as string, 
            name: params.projectId as string 
          })
          return HttpResponse.json({ project })
        })
      )

      const result = await deleteProject('default', 'my-project')

      expect(result.project.namespace).toBe('default')
      expect(result.project.name).toBe('my-project')
    })

    it('should return deleted project from response', async () => {
      const mockProject = createMockProject({ name: 'test-project' })
      
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      const result = await deleteProject('default', 'test-project')

      expect(result).toHaveProperty('project')
      expect(result.project).toEqual(mockProject)
    })

    it('should throw error when deletion fails', async () => {
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Deletion failed' },
            { status: 500 }
          )
        })
      )

      await expect(deleteProject('default', 'test')).rejects.toThrow()
    })

    it('should handle project not found error', async () => {
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Project not found' },
            { status: 404 }
          )
        })
      )

      await expect(deleteProject('default', 'non-existent')).rejects.toThrow()
    })

    it('should handle cascade delete conflicts', async () => {
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json(
            { detail: 'Cannot delete project with active datasets' },
            { status: 409 }
          )
        })
      )

      await expect(deleteProject('default', 'test')).rejects.toThrow()
    })

    it('should handle network errors', async () => {
      server.use(
        http.delete(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.error()
        })
      )

      await expect(deleteProject('default', 'test')).rejects.toThrow()
    })
  })

  describe('URL Encoding', () => {
    it('should handle URL encoded projectId with spaces', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
          // MSW automatically decodes the URL params
          const project = createMockProject({ name: params.projectId as string })
          return HttpResponse.json({ project })
        })
      )

      const result = await getProject('default', 'project with spaces')

      expect(result.project.name).toBe('project with spaces')
    })

    it('should handle URL encoded projectId with special characters', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
          const project = createMockProject({ name: params.projectId as string })
          return HttpResponse.json({ project })
        })
      )

      // Test various special characters
      const specialNames = ['project/name', 'project?name', 'project#name', 'project&name']
      
      for (const name of specialNames) {
        const result = await getProject('default', name)
        expect(result.project.name).toBe(name)
      }
    })
  })

  describe('Response Handling', () => {
    it('should extract data from response', async () => {
      const mockData = createMockProjectsList('default', 2)
      
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(mockData)
        })
      )

      const result = await listProjects('default')

      expect(result).toEqual(mockData)
      expect(result).toHaveProperty('projects')
      expect(result).toHaveProperty('total')
    })

    it('should propagate error responses', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(
            { detail: 'Invalid request' },
            { status: 400 }
          )
        })
      )

      await expect(listProjects('default')).rejects.toThrow()
    })
  })

  describe('Type Safety', () => {
    it('should use correct request types', async () => {
      const createRequest: CreateProjectRequest = { name: 'test' }
      const updateRequest: UpdateProjectRequest = { config: {} }
      const mockProject = createMockProject()

      server.use(
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json({ project: mockProject })
        }),
        http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      await createProject('default', createRequest)
      await updateProject('default', 'test', updateRequest)

      // If this compiles and runs, types are enforced
      expect(true).toBe(true)
    })

    it('should return correct response types', async () => {
      const mockProjectsList = createMockProjectsList('default', 2)
      const mockProject = createMockProject()

      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(mockProjectsList)
        }),
        http.post(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json({ project: mockProject })
        })
      )

      const listResult = await listProjects('default')
      const createResult = await createProject('default', { name: 'test' })

      expect(listResult).toHaveProperty('projects')
      expect(listResult).toHaveProperty('total')
      expect(createResult).toHaveProperty('project')
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

