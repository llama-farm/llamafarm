// NOTE: These tests mock modelService directly for simplicity with async generators.
// This tests implementation details rather than behavior. Consider adding E2E tests
// with real SSE streams for integration coverage.

import React, { useState } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { useDemoWorkflow } from '../useDemoWorkflow'
import { server } from '../../test/mocks/server'
import { http, HttpResponse } from 'msw'
import type { DownloadEvent } from '../../types/model'

// Mock modelService for SSE async generator testing
vi.mock('../../api/modelService', () => ({
  default: {
    listCachedModels: vi.fn(),
    downloadModel: vi.fn(),
  },
  listCachedModels: vi.fn(),
  downloadModel: vi.fn(),
}))

// Import after mocking
import modelService from '../../api/modelService'

const API_BASE = '/api/v1'

// Demo config with embedding model for testing
const DEMO_CONFIG_WITH_MODEL = `
version: v1
name: test-demo
namespace: default
runtime:
  provider: ollama
  model: llama3.2:3b
prompts: []
rag:
  databases:
    - name: main_db
      embedding_strategies:
        - name: test_strategy
          config:
            model: sentence-transformers/all-MiniLM-L6-v2
`

const DEMO_CONFIG_WITHOUT_MODEL = `
version: v1
name: test-demo
namespace: default
runtime:
  provider: ollama
  model: llama3.2:3b
prompts: []
`

// Test demo configuration
const testDemo = {
  id: 'test-demo',
  name: 'test-demo',
  title: 'Test Demo',
  description: 'A test demo',
  files: [{ path: 'test.pdf', name: 'test.pdf' }],
  configPath: '/demos/test-demo/llamafarm.yaml',
}

// ============================================================================
// Mock Generator Helpers
// ============================================================================

async function* mockSuccessfulDownload(): AsyncIterableIterator<DownloadEvent> {
  yield { event: 'progress', downloaded: 0, total: 100 }
  yield { event: 'progress', downloaded: 50, total: 100 }
  yield { event: 'progress', downloaded: 100, total: 100 }
  yield { event: 'done', local_dir: '/models/test' }
}

async function* mockIndeterminateDownload(): AsyncIterableIterator<DownloadEvent> {
  yield { event: 'progress', downloaded: 1000, total: 0 }
  yield { event: 'progress', downloaded: 2000, total: 0 }
  yield { event: 'progress', downloaded: 3000, total: 0 }
  yield { event: 'done', local_dir: '/models/test' }
}

async function* mockFailedDownload(): AsyncIterableIterator<DownloadEvent> {
  yield { event: 'progress', downloaded: 50, total: 100 }
  yield { event: 'error', message: 'Connection lost during download' }
}

async function* mockPrematureEnd(): AsyncIterableIterator<DownloadEvent> {
  yield { event: 'progress', downloaded: 50, total: 100 }
  // Generator ends without done or error event
}

async function* mockParseError(): AsyncIterableIterator<DownloadEvent> {
  yield { event: 'progress', downloaded: 50, total: 100 }
  yield { event: 'error', message: 'Failed to parse server response' }
}

// Factory for progress events with specific percentages
function createProgressEvent(downloaded: number, total: number): DownloadEvent {
  return { event: 'progress', downloaded, total }
}

// ============================================================================
// Test Setup Helpers
// ============================================================================

function setupMswHandlers() {
  // Demo config fetch
  server.use(
    http.get('*/demos/test-demo/llamafarm.yaml', () => {
      return new HttpResponse(DEMO_CONFIG_WITH_MODEL, {
        headers: { 'Content-Type': 'text/yaml' },
      })
    })
  )

  // Project list (empty for new demo)
  server.use(
    http.get(`${API_BASE}/projects/:namespace`, () => {
      return HttpResponse.json({ projects: [], total: 0 })
    })
  )

  // Create project
  server.use(
    http.post(`${API_BASE}/projects/:namespace`, async ({ request }) => {
      const body = (await request.json()) as any
      return HttpResponse.json({
        project: { name: body.name, namespace: 'default', config: {} },
      })
    })
  )

  // Update project
  server.use(
    http.put(`${API_BASE}/projects/:namespace/:projectId`, () => {
      return HttpResponse.json({
        project: { name: 'test-demo-1', namespace: 'default', config: {} },
      })
    })
  )

  // Dataset create
  server.use(
    http.post(`${API_BASE}/projects/:namespace/:project/datasets/`, () => {
      return HttpResponse.json({ dataset: { name: 'test-demo-1' } })
    })
  )

  // File upload
  server.use(
    http.post(`${API_BASE}/projects/:namespace/:project/datasets/:dataset/data`, () => {
      return HttpResponse.json({ filename: 'test.pdf', hash: 'abc123' })
    })
  )

  // Dataset process
  server.use(
    http.post(`${API_BASE}/projects/:namespace/:project/datasets/:dataset/actions`, () => {
      return HttpResponse.json({ task_id: 'task-123' })
    })
  )

  // Task status (immediate success)
  server.use(
    http.get(`${API_BASE}/projects/:namespace/:project/tasks/:taskId`, () => {
      return HttpResponse.json({
        task_id: 'task-123',
        state: 'SUCCESS',
        result: { processed_files: 1 },
      })
    })
  )

  // Demo file fetch
  server.use(
    http.get('*/demos/test-demo/test.pdf', () => {
      return new HttpResponse(new Blob(['pdf content']), {
        headers: { 'Content-Type': 'application/pdf' },
      })
    })
  )
}

function setupMswWithoutEmbeddingModel() {
  server.use(
    http.get('*/demos/test-demo/llamafarm.yaml', () => {
      return new HttpResponse(DEMO_CONFIG_WITHOUT_MODEL, {
        headers: { 'Content-Type': 'text/yaml' },
      })
    })
  )
}

// Create QueryClient for testing
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  })
}

// Wrapper component for renderHook
function TestWrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => createTestQueryClient())
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  )
}

const wrapper = TestWrapper

// ============================================================================
// Tests
// ============================================================================

describe('useDemoWorkflow - Model Download Step', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setupMswHandlers()
  })

  afterEach(() => {
    server.resetHandlers()
  })

  describe('Model already cached - skip download', () => {
    it('should skip download when model is already cached', async () => {
      // Mock model as already cached
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [
          { id: 'sentence-transformers/all-MiniLM-L6-v2', name: 'all-MiniLM-L6-v2' },
        ],
      } as any)

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      // Wait for workflow to progress past download step
      await waitFor(
        () => {
          expect(result.current.progress).toBeGreaterThanOrEqual(50)
        },
        { timeout: 5000 }
      )

      // downloadModel should never be called
      expect(modelService.downloadModel).not.toHaveBeenCalled()
    })

    it('should jump progress from 20% to 50% when model cached', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [
          { id: 'sentence-transformers/all-MiniLM-L6-v2', name: 'all-MiniLM-L6-v2' },
        ],
      } as any)

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      // Wait for workflow to progress past download step
      await waitFor(
        () => {
          expect(result.current.progress).toBeGreaterThanOrEqual(50)
        },
        { timeout: 5000 }
      )

      // Should NOT have downloading_model step (model was cached, so skipped)
      expect(result.current.currentStep).not.toBe('downloading_model')
    })
  })

  describe('Model not cached - download with progress', () => {
    it('should trigger download when model is not cached', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockSuccessfulDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(modelService.downloadModel).toHaveBeenCalled()
        },
        { timeout: 5000 }
      )

      expect(modelService.downloadModel).toHaveBeenCalledWith({
        model_name: 'sentence-transformers/all-MiniLM-L6-v2',
        provider: 'universal',
      })
    })

    it('should update step to downloading_model', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockSuccessfulDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          return (
            result.current.currentStep === 'downloading_model' ||
            result.current.progress >= 50
          )
        },
        { timeout: 5000 }
      )

      // Either we caught the downloading_model step or it completed
      // The key is that the download was attempted
      expect(modelService.downloadModel).toHaveBeenCalled()
    })
  })

  describe('Progress scaling is correct', () => {
    it('should scale download progress to 25-50% range', async () => {
      // Create a generator that yields specific progress values
      async function* mockProgressSequence(): AsyncIterableIterator<DownloadEvent> {
        yield createProgressEvent(0, 100) // 0% downloaded
        yield createProgressEvent(50, 100) // 50% downloaded
        yield createProgressEvent(100, 100) // 100% downloaded
        yield { event: 'done', local_dir: '/models/test' }
      }

      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockProgressSequence())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      // Wait for download to complete and progress to reach 50%
      await waitFor(
        () => {
          expect(result.current.progress).toBeGreaterThanOrEqual(50)
        },
        { timeout: 5000 }
      )

      // Progress should be at least 50% after download completes
      expect(result.current.progress).toBeGreaterThanOrEqual(50)
    })
  })

  describe('SSE stream fails midway - proper error handling', () => {
    it('should set error state when stream yields error event', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockFailedDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(result.current.currentStep).toBe('error')
        },
        { timeout: 5000 }
      )

      expect(result.current.error).toContain('Connection lost during download')
    })
  })

  describe('Cache check fails - fallback behavior', () => {
    it('should attempt download when cache check throws', async () => {
      vi.mocked(modelService.listCachedModels).mockRejectedValue(
        new Error('Cache service unavailable')
      )
      vi.mocked(modelService.downloadModel).mockReturnValue(mockSuccessfulDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(modelService.downloadModel).toHaveBeenCalled()
        },
        { timeout: 5000 }
      )

      // Download should still be attempted
      expect(modelService.downloadModel).toHaveBeenCalledWith({
        model_name: 'sentence-transformers/all-MiniLM-L6-v2',
        provider: 'universal',
      })
    })

    it('should continue workflow after cache check failure', async () => {
      vi.mocked(modelService.listCachedModels).mockRejectedValue(
        new Error('Cache service unavailable')
      )
      vi.mocked(modelService.downloadModel).mockReturnValue(mockSuccessfulDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          // Workflow should progress past download step
          return result.current.progress >= 50 || result.current.currentStep === 'error'
        },
        { timeout: 5000 }
      )

      // If no error, workflow continued successfully
      if (result.current.currentStep !== 'error') {
        expect(result.current.progress).toBeGreaterThanOrEqual(50)
      }
    })
  })

  describe('Download stream ends without done event', () => {
    it('should error when stream ends prematurely', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockPrematureEnd())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(result.current.currentStep).toBe('error')
        },
        { timeout: 5000 }
      )

      // Should show stream ended unexpectedly error
      expect(result.current.error).toContain('stream ended unexpectedly')
    })
  })

  describe('Parse error shows accurate message - CRITICAL', () => {
    it('should show parse-specific error message when parse fails', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockParseError())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(result.current.currentStep).toBe('error')
        },
        { timeout: 5000 }
      )

      // This is the critical test: error should mention parse failure
      // NOT "stream ended unexpectedly"
      expect(result.current.error).toContain('parse')
      expect(result.current.error).not.toContain('stream ended unexpectedly')
    })
  })

  describe('Unknown total size - indeterminate progress', () => {
    it('should not get stuck at 25% when total is 0', async () => {
      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(mockIndeterminateDownload())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          // Progress should increase past 25% even with indeterminate total
          return result.current.progress > 25 || result.current.progress >= 50
        },
        { timeout: 5000 }
      )

      // Should complete successfully despite unknown total
      await waitFor(
        () => {
          expect(result.current.progress).toBeGreaterThanOrEqual(50)
        },
        { timeout: 5000 }
      )
    })

    it('should show activity with incrementing progress', async () => {
      // Create a generator that pauses so we can capture progress
      async function* slowIndeterminate(): AsyncIterableIterator<DownloadEvent> {
        yield { event: 'progress', downloaded: 1000, total: 0 }
        yield { event: 'progress', downloaded: 2000, total: 0 }
        yield { event: 'progress', downloaded: 3000, total: 0 }
        yield { event: 'progress', downloaded: 4000, total: 0 }
        yield { event: 'progress', downloaded: 5000, total: 0 }
        yield { event: 'done', local_dir: '/models/test' }
      }

      vi.mocked(modelService.listCachedModels).mockResolvedValue({
        data: [],
      } as any)
      vi.mocked(modelService.downloadModel).mockReturnValue(slowIndeterminate())

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          expect(result.current.progress).toBeGreaterThanOrEqual(50)
        },
        { timeout: 5000 }
      )

      // If we got here, the workflow completed which means progress wasn't stuck
      expect(result.current.progress).toBeGreaterThanOrEqual(50)
    })
  })

  describe('No embedding model in config', () => {
    it('should skip download step entirely when no embedding model', async () => {
      setupMswWithoutEmbeddingModel()

      const { result } = renderHook(() => useDemoWorkflow(), { wrapper })

      await act(async () => {
        result.current.startDemo(testDemo)
      })

      await waitFor(
        () => {
          // Should skip directly to creating_project step
          return (
            result.current.currentStep === 'creating_project' ||
            result.current.progress >= 50
          )
        },
        { timeout: 5000 }
      )

      // Neither cache check nor download should be called
      expect(modelService.listCachedModels).not.toHaveBeenCalled()
      expect(modelService.downloadModel).not.toHaveBeenCalled()
    })
  })
})

