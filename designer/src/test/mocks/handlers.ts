import { http, HttpResponse } from 'msw'
import {
  createMockProject,
  createMockProjectsList,
} from '../factories/projectFactory'
import {
  createMockDatasetsList,
} from '../factories/datasetFactory'
import {
  createMockEmbeddingStrategies,
  createMockDatabases,
} from '../factories/embeddingStrategyFactory'

const API_BASE = '/api/v1'

/**
 * MSW Request Handlers for API mocking
 * These handlers intercept API calls during tests and return mock data
 */
export const handlers = [
  // Projects endpoints
  http.get(`${API_BASE}/projects/:namespace`, ({ params }) => {
    const { namespace } = params
    return HttpResponse.json(createMockProjectsList(namespace as string))
  }),

  http.post(`${API_BASE}/projects/:namespace`, async ({ request, params }) => {
    const { namespace } = params
    const body = (await request.json()) as any
    const project = createMockProject({
      namespace: namespace as string,
      name: body.name,
      config: body.config || {},
    })
    return HttpResponse.json({ project })
  }),

  http.get(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
    const { namespace, projectId } = params
    const project = createMockProject({
      namespace: namespace as string,
      name: projectId as string,
    })
    return HttpResponse.json({ project })
  }),

  http.put(
    `${API_BASE}/projects/:namespace/:projectId`,
    async ({ request, params }) => {
      const { namespace, projectId } = params
      const body = (await request.json()) as any
      const project = createMockProject({
        namespace: namespace as string,
        name: projectId as string,
        config: body.config,
      })
      return HttpResponse.json({ project })
    }
  ),

  http.delete(`${API_BASE}/projects/:namespace/:projectId`, ({ params }) => {
    const { namespace, projectId } = params
    const project = createMockProject({
      namespace: namespace as string,
      name: projectId as string,
    })
    return HttpResponse.json({ project })
  }),

  // Datasets endpoints
  http.get(
    `${API_BASE}/projects/:namespace/:project/datasets/`,
    () => {
      return HttpResponse.json(createMockDatasetsList())
    }
  ),

  http.get(
    `${API_BASE}/projects/:namespace/:project/datasets/strategies`,
    () => {
      return HttpResponse.json({
        data_processing_strategies: createMockEmbeddingStrategies(),
        databases: createMockDatabases(),
      })
    }
  ),

  http.post(
    `${API_BASE}/projects/:namespace/:project/datasets/`,
    async ({ request }) => {
      const body = (await request.json()) as any
      const dataset = {
        name: body.name,
        data_processing_strategy: body.data_processing_strategy,
        database: body.database,
        files: [],
      }
      return HttpResponse.json({ dataset })
    }
  ),

  http.delete(
    `${API_BASE}/projects/:namespace/:project/datasets/:dataset`,
    ({ params }) => {
      const { dataset } = params
      return HttpResponse.json({
        name: dataset,
        message: 'Dataset deleted successfully',
      })
    }
  ),

  http.post(
    `${API_BASE}/projects/:namespace/:project/datasets/:dataset/actions`,
    async () => {
      const taskId = `task-${Date.now()}`
      return HttpResponse.json({
        message: 'Accepted',
        task_uri: `/tasks/${taskId}`,
        task_id: taskId,
      })
    }
  ),

  http.post(
    `${API_BASE}/projects/:namespace/:project/datasets/:dataset/data`,
    async () => {
      return HttpResponse.json({
        filename: 'test-file.pdf',
        hash: `hash-${Date.now()}`,
        processed: false,
        skipped: false,
      })
    }
  ),

  http.delete(
    `${API_BASE}/projects/:namespace/:project/datasets/:dataset/data/:fileHash`,
    ({ params }) => {
      const { fileHash } = params
      return HttpResponse.json({
        file_hash: fileHash,
        message: 'File deleted successfully',
      })
    }
  ),

  // Task status endpoint (matches TaskStatusResponse interface)
  http.get(
    `${API_BASE}/projects/:namespace/:project/tasks/:taskId`,
    ({ params }) => {
      const { taskId } = params
      return HttpResponse.json({
        task_id: taskId,
        state: 'SUCCESS',
        meta: null,
        result: { processed_files: 10, failed_files: 0, skipped_files: 0 },
        error: null,
        traceback: null,
      })
    }
  ),

  // Health check endpoint
  http.get(`${API_BASE}/health`, () => {
    return HttpResponse.json({
      status: 'healthy',
      version: '1.0.0',
    })
  }),

  // System info endpoint
  http.get(`${API_BASE}/system/info`, () => {
    return HttpResponse.json({
      version: '1.0.0',
      environment: 'test',
    })
  }),

  // Version check endpoint
  http.get(`${API_BASE}/system/version-check`, () => {
    return HttpResponse.json({
      current_version: '1.0.0',
      latest_version: '1.0.0',
      upgrade_available: false,
      release_notes_url: null,
    })
  }),

  // Examples endpoints
  http.get(`${API_BASE}/examples/datasets`, () => {
    return HttpResponse.json({
      datasets: [
        { id: 'example-1', name: 'Example Dataset 1' },
        { id: 'example-2', name: 'Example Dataset 2' },
      ],
    })
  }),

  http.get(`${API_BASE}/examples/:exampleId/datasets`, () => {
    return HttpResponse.json({
      datasets: [
        { id: 'dataset-1', name: 'Dataset 1' },
        { id: 'dataset-2', name: 'Dataset 2' },
      ],
    })
  }),

  http.post(
    `${API_BASE}/examples/:exampleId/import-dataset`,
    async ({ request }) => {
      const body = (await request.json()) as any
      return HttpResponse.json({
        project: body.project || 'test-project',
        namespace: body.namespace || 'default',
        dataset: body.dataset || 'test-dataset',
        file_count: 5,
        task_id: `task-${Date.now()}`,
      })
    }
  ),
]

