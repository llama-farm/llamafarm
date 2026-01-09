import { useCallback } from 'react'
import { useDevToolsOptional, type CapturedRequest } from '../contexts/DevToolsContext'

// Generate a unique ID for requests
function generateId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

// Extract headers from a Headers object or record
function headersToRecord(headers: Headers | Record<string, string>): Record<string, string> {
  if (headers instanceof Headers) {
    const record: Record<string, string> = {}
    headers.forEach((value, key) => {
      record[key] = value
    })
    return record
  }
  return headers
}

export interface DevToolsCaptureCallbacks {
  onRequestStart: (request: Omit<CapturedRequest, 'streamChunks' | 'streamComplete'>) => void
  onStreamChunk: (id: string, chunk: any) => void
  onStreamComplete: (id: string) => void
  onResponse: (id: string, response: {
    status: number
    statusText: string
    headers: Record<string, string>
    body: any
    requestId?: string | null
  }) => void
  onError: (id: string, error: string) => void
}

/**
 * Hook to get DevTools capture callbacks
 * Returns undefined if DevTools context is not available
 */
export function useDevToolsCapture(): DevToolsCaptureCallbacks | undefined {
  const devTools = useDevToolsOptional()

  const onRequestStart = useCallback(
    (request: Omit<CapturedRequest, 'streamChunks' | 'streamComplete'>) => {
      devTools?.captureRequest(request)
    },
    [devTools]
  )

  const onStreamChunk = useCallback(
    (id: string, chunk: any) => {
      devTools?.addStreamChunk(id, chunk)
    },
    [devTools]
  )

  const onStreamComplete = useCallback(
    (id: string) => {
      devTools?.completeStream(id)
    },
    [devTools]
  )

  const onResponse = useCallback(
    (id: string, response: {
      status: number
      statusText: string
      headers: Record<string, string>
      body: any
      requestId?: string | null
    }) => {
      devTools?.updateResponse(id, response)
    },
    [devTools]
  )

  const onError = useCallback(
    (id: string, error: string) => {
      devTools?.setError(id, error)
    },
    [devTools]
  )

  if (!devTools) {
    return undefined
  }

  return {
    onRequestStart,
    onStreamChunk,
    onStreamComplete,
    onResponse,
    onError,
  }
}

/**
 * Create a captured request object for DevTools
 */
export function createCapturedRequest(
  method: CapturedRequest['method'],
  url: string,
  fullUrl: string,
  headers: Record<string, string>,
  body: any,
  isStreaming: boolean
): Omit<CapturedRequest, 'streamChunks' | 'streamComplete'> {
  return {
    id: generateId(),
    requestId: null,
    method,
    url,
    fullUrl,
    headers,
    body,
    timestamp: Date.now(),
    isStreaming,
  }
}

/**
 * Extract request ID from response headers
 */
export function extractRequestId(headers: Headers | Record<string, string>): string | null {
  if (headers instanceof Headers) {
    return headers.get('x-request-id') || headers.get('X-Request-ID') || null
  }
  return headers['x-request-id'] || headers['X-Request-ID'] || null
}

export { generateId, headersToRecord }
