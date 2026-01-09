/**
 * Unified Chat Completions API Service
 *
 * This service provides a single interface for all chat completion operations.
 * Supports both streaming and non-streaming modes, with session management.
 *
 * Usage:
 * - Project Chat: Use current project's namespace and project ID
 * - Dev Chat: Use hardcoded "llamafarm" namespace and "project_seed" project ID
 */

import { apiClient } from './client'
import {
  ChatRequest,
  ChatResponse,
  ChatStreamChunk,
  StreamingChatOptions,
  NetworkError,
  ValidationError,
  ChatApiError,
} from '../types/chat'
import { handleSSEResponse } from '../utils/sseUtils'
import {
  type DevToolsCaptureCallbacks,
  createCapturedRequest,
  extractRequestId,
  headersToRecord,
} from '../hooks/useDevToolsCapture'

/**
 * Result from non-streaming chat completion
 */
export interface ChatCompletionResult {
  response: ChatResponse
  sessionId: string
}

/**
 * Send a non-streaming chat completion request
 *
 * @param namespace - The project namespace
 * @param projectId - The project ID
 * @param request - The chat request payload
 * @param sessionId - Optional session ID for conversation continuity
 * @returns Promise<ChatCompletionResult> - Response and server-provided session ID
 */
export async function sendChatCompletion(
  namespace: string,
  projectId: string,
  request: ChatRequest,
  sessionId?: string,
  devToolsCapture?: DevToolsCaptureCallbacks
): Promise<ChatCompletionResult> {
  // Validate inputs
  if (!namespace || !projectId) {
    throw new ValidationError('Namespace and project ID are required', {
      namespace,
      projectId,
    })
  }

  if (!request.messages || request.messages.length === 0) {
    throw new ValidationError('At least one message is required', {
      messages: request.messages,
    })
  }

  // Validate message content
  for (const message of request.messages) {
    if (!message.content || typeof message.content !== 'string') {
      throw new ValidationError('All messages must have valid string content', {
        message,
      })
    }
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }

  // Add session header if provided
  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  // Ensure stream is false for non-streaming requests
  const chatRequest = { ...request, stream: false }

  // Build full URL for DevTools capture
  const rawBaseURL = apiClient.defaults.baseURL || ''
  const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL.slice(0, -1) : rawBaseURL
  const urlPath = `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}/chat/completions`
  const fullUrl = `${baseURL}${urlPath}`

  // Capture request for DevTools
  let capturedRequestId: string | undefined
  if (devToolsCapture) {
    const captured = createCapturedRequest('POST', urlPath, fullUrl, headers, chatRequest, false)
    capturedRequestId = captured.id
    devToolsCapture.onRequestStart(captured)
  }

  try {
    const response = await apiClient.post<ChatResponse>(urlPath, chatRequest, { headers })

    // Extract session ID from response headers (server provides this)
    const responseSessionId =
      response.headers['x-session-id'] ||
      response.headers['X-Session-ID'] ||
      sessionId || // Fallback to sent session ID
      ''

    if (!responseSessionId) {
      console.warn('No session ID received from server response')
    }

    // Capture response for DevTools
    if (devToolsCapture && capturedRequestId) {
      devToolsCapture.onResponse(capturedRequestId, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers as Record<string, string>,
        body: response.data,
        requestId: response.headers['x-request-id'] || response.headers['X-Request-ID'] || null,
      })
    }

    return {
      response: response.data,
      sessionId: responseSessionId,
    }
  } catch (error) {
    // Capture error for DevTools
    if (devToolsCapture && capturedRequestId) {
      devToolsCapture.onError(
        capturedRequestId,
        error instanceof Error ? error.message : 'Unknown error'
      )
    }
    if (
      error instanceof ValidationError ||
      error instanceof NetworkError ||
      error instanceof ChatApiError
    ) {
      throw error
    }

    throw new NetworkError(
      `Failed to send chat completion: ${error instanceof Error ? error.message : 'Unknown error'}`,
      error instanceof Error ? error : new Error('Unknown error')
    )
  }
}

/**
 * Send a streaming chat completion request using Server-Sent Events
 *
 * @param namespace - The project namespace
 * @param projectId - The project ID
 * @param request - The chat request payload (stream will be set to true)
 * @param sessionId - Optional session ID for conversation continuity
 * @param options - Streaming options with callbacks
 * @param activeProject - Optional active project for dev chat context (format: "namespace/project")
 * @returns Promise<string> - The final session ID
 */
export async function streamChatCompletion(
  namespace: string,
  projectId: string,
  request: ChatRequest,
  sessionId?: string,
  options: StreamingChatOptions = {},
  activeProject?: string,
  devToolsCapture?: DevToolsCaptureCallbacks
): Promise<string> {
  const { onChunk, onError, onComplete, signal } = options

  // Validate inputs
  if (!namespace || !projectId) {
    const error = new ValidationError('Namespace and project ID are required', {
      namespace,
      projectId,
    })
    onError?.(error)
    throw error
  }

  if (!request.messages || request.messages.length === 0) {
    const error = new ValidationError('At least one message is required', {
      messages: request.messages,
    })
    onError?.(error)
    throw error
  }

  // Validate message content
  for (const message of request.messages) {
    if (!message.content || typeof message.content !== 'string') {
      const error = new ValidationError(
        'All messages must have valid string content',
        { message }
      )
      onError?.(error)
      throw error
    }
  }

  // Ensure streaming is enabled
  const streamingRequest = { ...request, stream: true }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
    'Cache-Control': 'no-cache',
  }

  // Add session header if provided
  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  // Add active project header if provided (for dev chat context)
  if (activeProject) {
    headers['X-Active-Project'] = activeProject
  }

  // Use fetch directly for streaming instead of axios
  const rawBaseURL = apiClient.defaults.baseURL || ''
  const baseURL = rawBaseURL.endsWith('/') ? rawBaseURL.slice(0, -1) : rawBaseURL
  const urlPath = `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}/chat/completions`
  const url = `${baseURL}${urlPath}`

  // Capture request for DevTools
  let capturedRequestId: string | undefined
  if (devToolsCapture) {
    const captured = createCapturedRequest('POST', urlPath, url, headers, streamingRequest, true)
    capturedRequestId = captured.id
    devToolsCapture.onRequestStart(captured)
  }

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      body: JSON.stringify(streamingRequest),
      signal,
    })

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error')
      const error = new NetworkError(
        `HTTP ${response.status}: ${response.statusText} - ${errorText}`,
        new Error(`Fetch failed with status ${response.status}`)
      )
      // Capture error for DevTools
      if (devToolsCapture && capturedRequestId) {
        devToolsCapture.onError(capturedRequestId, error.message)
      }
      throw error
    }

    // Extract session ID from response headers (server provides this)
    const responseSessionId =
      response.headers.get('x-session-id') ||
      response.headers.get('X-Session-ID') ||
      sessionId || // Fallback to sent session ID
      ''

    if (!responseSessionId) {
      console.warn('No session ID received from streaming response headers')
    }

    // Capture response headers for DevTools (before streaming starts)
    if (devToolsCapture && capturedRequestId) {
      const serverRequestId = extractRequestId(response.headers)
      // Update with response headers - body will be filled via stream chunks
      devToolsCapture.onResponse(capturedRequestId, {
        status: response.status,
        statusText: response.statusText,
        headers: headersToRecord(response.headers),
        body: null, // Will be populated via stream chunks
        requestId: serverRequestId,
      })
    }

    // Handle the streaming response using SSE utility
    await handleSSEResponse<ChatStreamChunk>(
      response,
      chunk => {
        // Capture chunk for DevTools
        if (devToolsCapture && capturedRequestId) {
          devToolsCapture.onStreamChunk(capturedRequestId, chunk)
        }
        onChunk?.(chunk)
      },
      {
        signal,
        onComplete: () => {
          // Mark stream as complete for DevTools
          if (devToolsCapture && capturedRequestId) {
            devToolsCapture.onStreamComplete(capturedRequestId)
          }
          onComplete?.()
        },
        onError: error => {
          console.error(
            'onError callback invoked in streamChatCompletion:',
            error
          )
          // Capture error for DevTools
          if (devToolsCapture && capturedRequestId) {
            devToolsCapture.onError(capturedRequestId, error.message)
          }
          onError?.(error)
        },
      }
    )

    return responseSessionId
  } catch (error) {
    // Capture error for DevTools
    if (devToolsCapture && capturedRequestId) {
      devToolsCapture.onError(
        capturedRequestId,
        error instanceof Error ? error.message : 'Unknown error'
      )
    }

    // Handle abort errors specifically
    if (error instanceof Error && error.name === 'AbortError') {
      const abortError = new NetworkError(
        'Chat completion streaming request was cancelled',
        error
      )
      onError?.(abortError)
      throw abortError
    }

    const networkError =
      error instanceof NetworkError
        ? error
        : new NetworkError(
            error instanceof Error
              ? error.message
              : 'Unknown chat completion streaming error',
            error instanceof Error ? error : new Error('Unknown error')
          )

    onError?.(networkError)
    throw networkError
  }
}

/**
 * Helper function to create a simple chat request from a message string
 *
 * @param message - The user message content
 * @param options - Additional request options
 * @returns ChatRequest - The formatted request
 */
export function createChatCompletionRequest(
  message: string,
  options: Partial<ChatRequest> = {}
): ChatRequest {
  // Validate message
  if (typeof message !== 'string' || !message.trim()) {
    throw new ValidationError('Message must be a non-empty string', { message })
  }

  return {
    messages: [{ role: 'user', content: message.trim() }],
    ...options,
  }
}

export default {
  sendChatCompletion,
  streamChatCompletion,
  createChatCompletionRequest,
}
