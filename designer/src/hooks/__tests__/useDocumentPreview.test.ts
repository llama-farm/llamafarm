/**
 * Tests for useDocumentPreview hook - TDD Red Phase
 * All tests written FIRST and will fail until implementation is complete.
 */

import React, { useState } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useDocumentPreview } from '../useDocumentPreview'
import { server } from '../../test/mocks/server'
import { http, HttpResponse } from 'msw'

const API_BASE = 'http://localhost:8000/v1'

const mockPreviewResponse = {
  original_text: 'This is test content for the preview.',
  chunks: [
    {
      chunk_index: 0,
      content: 'This is test content',
      start_position: 0,
      end_position: 20,
      char_count: 20,
      word_count: 4,
    },
    {
      chunk_index: 1,
      content: 'for the preview.',
      start_position: 15,
      end_position: 31,
      char_count: 16,
      word_count: 3,
    },
  ],
  filename: 'test.txt',
  size_bytes: 31,
  content_type: 'text/plain',
  parser_used: 'TextParser_Python',
  chunk_strategy: 'characters',
  chunk_size: 20,
  chunk_overlap: 5,
  total_chunks: 2,
  avg_chunk_size: 18.0,
  total_size_with_overlaps: 36,
  avg_overlap_size: 5.0,
  warnings: [],
}

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  })
}

function createWrapper() {
  return function TestWrapper({ children }: { children: React.ReactNode }) {
    const [queryClient] = useState(() => createTestQueryClient())
    return React.createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

describe('useDocumentPreview', () => {
  beforeEach(() => {
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        return HttpResponse.json(mockPreviewResponse)
      })
    )
  })

  afterEach(() => {
    server.resetHandlers()
  })

  it('calls preview API with correct parameters', async () => {
    let capturedRequest: any = null

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async ({ request, params }) => {
        capturedRequest = {
          params,
          body: await request.json(),
        }
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
      })
    })

    await waitFor(() => {
      expect(capturedRequest).not.toBeNull()
    })

    expect(capturedRequest.params.namespace).toBe('test-ns')
    expect(capturedRequest.params.project).toBe('test-project')
    expect(capturedRequest.params.database).toBe('default')
    expect(capturedRequest.body.file_hash).toBe('abc123')
  })

  it('returns loading state during request', async () => {
    // Add delay to see loading state
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    act(() => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
      })
    })

    // Should be loading
    expect(result.current.isPending).toBe(true)

    // Wait for completion
    await waitFor(() => {
      expect(result.current.isPending).toBe(false)
    })
  })

  it('returns data on success', async () => {
    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
      })
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(result.current.data).toEqual(mockPreviewResponse)
    expect(result.current.data?.original_text).toBe('This is test content for the preview.')
    expect(result.current.data?.chunks).toHaveLength(2)
  })

  it('returns error on failure', async () => {
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        return HttpResponse.json(
          { detail: 'File not found' },
          { status: 404 }
        )
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'nonexistent',
      })
    })

    await waitFor(() => {
      expect(result.current.isError).toBe(true)
    })

    expect(result.current.error).toBeDefined()
  })

  it('supports parameter overrides', async () => {
    let capturedBody: any = null

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
        chunk_size: 1000,
        chunk_overlap: 100,
        chunk_strategy: 'sentences',
      })
    })

    await waitFor(() => {
      expect(capturedBody).not.toBeNull()
    })

    expect(capturedBody.chunk_size).toBe(1000)
    expect(capturedBody.chunk_overlap).toBe(100)
    expect(capturedBody.chunk_strategy).toBe('sentences')
  })

  it('supports file content upload', async () => {
    let capturedBody: any = null

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    const fileContent = btoa('Hello World')

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_content: fileContent,
        filename: 'test.txt',
      })
    })

    await waitFor(() => {
      expect(capturedBody).not.toBeNull()
    })

    expect(capturedBody.file_content).toBe(fileContent)
    expect(capturedBody.filename).toBe('test.txt')
  })

  it('can be called multiple times', async () => {
    let callCount = 0

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        callCount++
        return HttpResponse.json({
          ...mockPreviewResponse,
          total_chunks: callCount,
        })
      })
    )

    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    // First call
    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
      })
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(result.current.data?.total_chunks).toBe(1)

    // Reset and call again
    await act(async () => {
      result.current.reset()
    })

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'def456',
      })
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(result.current.data?.total_chunks).toBe(2)
    expect(callCount).toBe(2)
  })
})

describe('useDocumentPreview - Type Safety', () => {
  beforeEach(() => {
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        return HttpResponse.json(mockPreviewResponse)
      })
    )
  })

  afterEach(() => {
    server.resetHandlers()
  })

  it('returns typed response data', async () => {
    const { result } = renderHook(
      () => useDocumentPreview('test-ns', 'test-project'),
      { wrapper: createWrapper() }
    )

    await act(async () => {
      result.current.mutate({
        database: 'default',
        file_hash: 'abc123',
      })
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    // Type assertions - these would fail at compile time if types are wrong
    const data = result.current.data!
    expect(typeof data.original_text).toBe('string')
    expect(Array.isArray(data.chunks)).toBe(true)
    expect(typeof data.chunks[0].chunk_index).toBe('number')
    expect(typeof data.chunks[0].content).toBe('string')
    expect(typeof data.total_chunks).toBe('number')
    expect(typeof data.avg_chunk_size).toBe('number')
  })
})
