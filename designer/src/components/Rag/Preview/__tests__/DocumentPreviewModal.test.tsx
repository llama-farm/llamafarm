/**
 * Tests for DocumentPreviewModal component - TDD Red Phase
 * All tests written FIRST and will fail until implementation is complete.
 */

import React, { useState } from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { DocumentPreviewModal } from '../DocumentPreviewModal'
import { server } from '../../../../test/mocks/server'
import { http, HttpResponse } from 'msw'

const API_BASE = 'http://localhost:8000/v1'

// Mock preview response
const mockPreviewResponse = {
  original_text: 'This is the original document text that spans multiple chunks.',
  chunks: [
    {
      chunk_index: 0,
      content: 'This is the original',
      start_position: 0,
      end_position: 20,
      char_count: 20,
      word_count: 4,
    },
    {
      chunk_index: 1,
      content: 'document text that spans',
      start_position: 15,
      end_position: 39,
      char_count: 24,
      word_count: 4,
    },
    {
      chunk_index: 2,
      content: 'multiple chunks.',
      start_position: 34,
      end_position: 50,
      char_count: 16,
      word_count: 2,
    },
  ],
  filename: 'test-document.txt',
  size_bytes: 50,
  content_type: 'text/plain',
  parser_used: 'TextParser_Python',
  chunk_strategy: 'characters',
  chunk_size: 25,
  chunk_overlap: 5,
  total_chunks: 3,
  avg_chunk_size: 20.0,
  total_size_with_overlaps: 60,
  avg_overlap_size: 5.0,
  warnings: [],
}

function setupMswHandlers() {
  server.use(
    http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
      return HttpResponse.json(mockPreviewResponse)
    }),
    // Handle CORS preflight
    http.options(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
      return new HttpResponse(null, { status: 204 })
    })
  )
}

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
      mutations: { retry: false },
    },
  })
}

function TestWrapper({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => createTestQueryClient())
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  )
}

describe('DocumentPreviewModal', () => {
  beforeEach(() => {
    setupMswHandlers()
  })

  afterEach(() => {
    server.resetHandlers()
  })

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    namespace: 'test-ns',
    project: 'test-project',
    database: 'default',
    fileHash: 'abc123',
  }

  it('renders when open', () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    expect(screen.getByTestId('preview-modal')).toBeInTheDocument()
    expect(screen.getByText(/Document Preview/i)).toBeInTheDocument()
  })

  it('does not render when closed', () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} isOpen={false} />
      </TestWrapper>
    )

    expect(screen.queryByTestId('preview-modal')).not.toBeInTheDocument()
  })

  it('displays statistics panel', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText(/Chunks: 3/)).toBeInTheDocument()
    })

    // Check for average size text
    await waitFor(() => {
      expect(screen.getByTestId('preview-stats')).toBeInTheDocument()
    })
  })

  it('fetches preview data on mount', async () => {
    const fetchSpy = vi.fn()
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        fetchSpy()
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalled()
    })
  })

  it('refetches when refresh button clicked', async () => {
    const user = userEvent.setup()
    let fetchCount = 0

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        fetchCount++
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    // Wait for initial fetch
    await waitFor(() => {
      expect(fetchCount).toBe(1)
    })

    // Wait for content to load
    await waitFor(() => {
      expect(screen.getByText(/Refresh Preview/i)).toBeInTheDocument()
    })

    // Click refresh
    const refreshButton = screen.getByRole('button', { name: /refresh/i })
    await user.click(refreshButton)

    await waitFor(() => {
      expect(fetchCount).toBe(2)
    })
  })

  it('shows loading state while fetching', async () => {
    // Delay response to see loading state
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async () => {
        await new Promise(resolve => setTimeout(resolve, 200))
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    // Should show loading state - wait a bit for component to mount and start fetching
    await waitFor(() => {
      expect(screen.getByTestId('preview-loading')).toBeInTheDocument()
    }, { timeout: 100 })

    // Wait for data to load
    await waitFor(() => {
      expect(screen.queryByTestId('preview-loading')).not.toBeInTheDocument()
    }, { timeout: 1000 })
  })

  it('shows error state on failure', async () => {
    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, () => {
        return HttpResponse.json(
          { detail: 'File not found' },
          { status: 404 }
        )
      })
    )

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('preview-error')).toBeInTheDocument()
    })
  })

  it('calls onClose when dialog closes', async () => {
    const onClose = vi.fn()

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} onClose={onClose} />
      </TestWrapper>
    )

    // Find and click the dialog close button (X button in Radix Dialog)
    const closeButton = screen.getByRole('button', { name: /close/i })
    fireEvent.click(closeButton)

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled()
    })
  })

  it('displays filename in header when data loads', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      // The filename is in a span within the title
      expect(screen.getByText(/test-document\.txt/)).toBeInTheDocument()
    })
  })

  it('displays parser information', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText(/TextParser_Python/)).toBeInTheDocument()
    })
  })

  it('passes override settings to API', async () => {
    const user = userEvent.setup()
    let capturedBody: any = null

    server.use(
      http.post(`${API_BASE}/projects/:namespace/:project/rag/databases/:database/preview`, async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json(mockPreviewResponse)
      })
    )

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    // Wait for initial load
    await waitFor(() => {
      expect(capturedBody).not.toBeNull()
    })

    // Wait for controls to be visible
    await waitFor(() => {
      expect(screen.getByText(/Refresh Preview/i)).toBeInTheDocument()
    })

    // Refresh to send new request
    const refreshButton = screen.getByRole('button', { name: /refresh/i })
    await user.click(refreshButton)

    await waitFor(() => {
      // Default chunk_size should be in the body
      expect(capturedBody.chunk_size).toBeDefined()
    })
  })

  it('clicking chunk in sidebar updates selection state', async () => {
    const user = userEvent.setup()

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    // Wait for content to load - check for sidebar chunk
    await waitFor(() => {
      expect(screen.getByTestId('sidebar-chunk-0')).toBeInTheDocument()
    }, { timeout: 5000 })

    // Click chunk in sidebar
    const sidebarChunk = screen.getByTestId('sidebar-chunk-1')
    await user.click(sidebarChunk)

    // Verify sidebar chunk gets selected styling
    await waitFor(() => {
      expect(sidebarChunk).toHaveClass('bg-blue-100')
    })
  })
})

describe('DocumentPreviewModal - Statistics Display', () => {
  beforeEach(() => {
    setupMswHandlers()
  })

  afterEach(() => {
    server.resetHandlers()
  })

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    namespace: 'test-ns',
    project: 'test-project',
    database: 'default',
    fileHash: 'abc123',
  }

  it('displays total chunks count', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText(/Chunks: 3/)).toBeInTheDocument()
    })
  })

  it('displays average chunk size', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      // Look within the stats panel
      const statsPanel = screen.getByTestId('preview-stats')
      expect(within(statsPanel).getByText(/Avg Size: 20 chars/)).toBeInTheDocument()
    })
  })

  it('displays total size with overlaps', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText(/Total with overlaps: 60/)).toBeInTheDocument()
    })
  })

  it('displays parser info', async () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    await waitFor(() => {
      expect(screen.getByText(/Parser: TextParser_Python/)).toBeInTheDocument()
    })
  })
})

describe('DocumentPreviewModal - Accessibility', () => {
  beforeEach(() => {
    setupMswHandlers()
  })

  afterEach(() => {
    server.resetHandlers()
  })

  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    namespace: 'test-ns',
    project: 'test-project',
    database: 'default',
    fileHash: 'abc123',
  }

  it('has accessible modal role', () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    expect(screen.getByRole('dialog')).toBeInTheDocument()
  })

  it('has proper dialog labelling', () => {
    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} />
      </TestWrapper>
    )

    const modal = screen.getByRole('dialog')
    // Radix Dialog uses aria-labelledby
    expect(modal).toHaveAttribute('aria-labelledby')
  })

  it('closes on escape key', async () => {
    const onClose = vi.fn()

    render(
      <TestWrapper>
        <DocumentPreviewModal {...defaultProps} onClose={onClose} />
      </TestWrapper>
    )

    // Focus the modal
    const modal = screen.getByRole('dialog')
    modal.focus()

    // Press escape
    fireEvent.keyDown(modal, { key: 'Escape' })

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled()
    })
  })
})
