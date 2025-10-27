import { apiClient } from './client'
import {
  ChatRequest,
  ChatResponse,
  DeleteSessionResponse,
  ChatStreamChunk,
  StreamingChatOptions,
  NetworkError,
  ValidationError,
} from '../types/chat'
import { handleSSEResponse } from '../utils/sseUtils'

/**
 * Send a chat message to the inference endpoint
 * @param chatRequest - The chat request payload
 * @param sessionId - Optional session ID for conversation continuity
 * @returns Promise<{data: ChatResponse, sessionId: string}>
 */
export async function chatInference(
  chatRequest: ChatRequest,
  sessionId?: string
): Promise<{ data: ChatResponse; sessionId: string }> {
  const headers: Record<string, string> = {}

  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  const response = await apiClient.post<ChatResponse>(
    '/inference/chat',
    chatRequest,
    {
      headers,
    }
  )

  // Extract session ID from response headers (server provides this)
  const responseSessionId = response.headers['x-session-id'] || sessionId || ''

  return {
    data: response.data,
    sessionId: responseSessionId,
  }
}

/**
 * Delete a chat session
 * @param sessionId - The session ID to delete
 * @returns Promise<DeleteSessionResponse>
 */
export async function deleteChatSession(
  sessionId: string
): Promise<DeleteSessionResponse> {
  const response = await apiClient.delete<DeleteSessionResponse>(
    `/inference/chat/sessions/${encodeURIComponent(sessionId)}`
  )
  return response.data
}

/**
 * Project-scoped chat (non-streaming): POST /v1/projects/{ns}/{project}/chat/completions
 */
export async function chatProject(
  namespace: string,
  project: string,
  chatRequest: ChatRequest,
  sessionId?: string
): Promise<{ data: ChatResponse; sessionId: string }> {
  const headers: Record<string, string> = {}
  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  const response = await apiClient.post<ChatResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/chat/completions`,
    chatRequest,
    { headers }
  )

  const responseSessionId = response.headers['x-session-id'] || sessionId || ''
  return { data: response.data, sessionId: responseSessionId }
}

/**
 * Project-scoped chat (streaming via SSE)
 */
export async function chatProjectStreaming(
  namespace: string,
  project: string,
  chatRequest: ChatRequest,
  sessionId?: string,
  options: StreamingChatOptions = {}
): Promise<string> {
  const { onChunk, onError, onComplete, signal } = options

  try {
    const streamingRequest = { ...chatRequest, stream: true }

    if (!streamingRequest.messages || streamingRequest.messages.length === 0) {
      throw new ValidationError('No messages in chat request', {
        messages: streamingRequest.messages,
      })
    }
    for (const message of streamingRequest.messages) {
      if (!message.content || typeof message.content !== 'string') {
        throw new ValidationError('Message content must be a string', {
          message,
        })
      }
    }

    const headers: Record<string, string> = {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
    }
    if (sessionId) {
      headers['X-Session-ID'] = sessionId
    }

    const rawBaseURL = apiClient.defaults.baseURL || ''
    const baseURL = rawBaseURL.endsWith('/')
      ? rawBaseURL.slice(0, -1)
      : rawBaseURL
    const url = `${baseURL}/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/chat/completions`

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
      throw new NetworkError(
        `HTTP ${response.status}: ${response.statusText}`,
        new Error(`Fetch failed with status ${response.status}`)
      )
    }

    const responseSessionId =
      response.headers.get('x-session-id') || sessionId || ''

    await handleSSEResponse<ChatStreamChunk>(
      response,
      chunk => onChunk?.(chunk),
      { signal, onComplete, onError }
    )

    return responseSessionId
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      const abortError = new NetworkError(
        'Streaming request was cancelled',
        error
      )
      onError?.(abortError)
      throw abortError
    }
    const networkError =
      error instanceof NetworkError
        ? error
        : new NetworkError(
            error instanceof Error ? error.message : 'Unknown streaming error',
            error instanceof Error ? error : new Error('Unknown error')
          )
    onError?.(networkError)
    throw networkError
  }
}

/**
 * Project-scoped session deletion
 */
export async function deleteProjectChatSession(
  namespace: string,
  project: string,
  sessionId: string
): Promise<DeleteSessionResponse> {
  const response = await apiClient.delete<DeleteSessionResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(project)}/chat/sessions/${encodeURIComponent(sessionId)}`
  )
  return response.data
}

// Helper functions for creating chat requests

/**
 * Create a simple chat request with a user message
 */
export function createChatRequest(
  message: string,
  options: Partial<ChatRequest> = {}
): ChatRequest {
  // Ensure message is a string and not empty
  if (typeof message !== 'string' || !message.trim()) {
    throw new Error('Message must be a non-empty string')
  }

  return {
    messages: [{ role: 'user', content: message.trim() }],
    metadata: {},
    modalities: [],
    response_format: {},
    stop: [],
    logit_bias: {},
    ...options,
  }
}

/**
 * Send a streaming chat message to the inference endpoint using Server-Sent Events
 * @param chatRequest - The chat request payload with stream: true
 * @param sessionId - Optional session ID for conversation continuity
 * @param options - Streaming options with callbacks
 * @returns Promise<string> - The final session ID
 */
export async function chatInferenceStreaming(
  chatRequest: ChatRequest,
  sessionId?: string,
  options: StreamingChatOptions = {}
): Promise<string> {
  const { onChunk, onError, onComplete, signal } = options

  try {
    // Ensure streaming is enabled and validate the request
    const streamingRequest = { ...chatRequest, stream: true }

    // Validate the request format
    if (!streamingRequest.messages || streamingRequest.messages.length === 0) {
      throw new ValidationError('No messages in chat request', {
        messages: streamingRequest.messages,
      })
    }

    // Ensure all messages have valid content
    for (const message of streamingRequest.messages) {
      if (!message.content || typeof message.content !== 'string') {
        throw new ValidationError('Message content must be a string', {
          message,
        })
      }
    }

    const headers: Record<string, string> = {
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
    }

    if (sessionId) {
      headers['X-Session-ID'] = sessionId
    }

    // Use fetch directly for streaming instead of axios
    // Construct the full URL using the same base as apiClient
    const baseURL = apiClient.defaults.baseURL || ''
    const url = `${baseURL}/inference/chat`

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
      throw new NetworkError(
        `HTTP ${response.status}: ${response.statusText}`,
        new Error(`Fetch failed with status ${response.status}`)
      )
    }

    // Extract session ID from response headers
    const responseSessionId =
      response.headers.get('x-session-id') || sessionId || ''

    // Handle the streaming response using SSE utility
    await handleSSEResponse<ChatStreamChunk>(
      response,
      chunk => onChunk?.(chunk),
      {
        signal,
        onComplete,
        onError,
      }
    )

    return responseSessionId
  } catch (error) {
    // Handle abort errors specifically
    if (error instanceof Error && error.name === 'AbortError') {
      const abortError = new NetworkError(
        'Streaming request was cancelled',
        error
      )
      onError?.(abortError)
      throw abortError
    }

    const networkError =
      error instanceof NetworkError
        ? error
        : new NetworkError(
            error instanceof Error ? error.message : 'Unknown streaming error',
            error instanceof Error ? error : new Error('Unknown error')
          )

    onError?.(networkError)
    throw networkError
  }
}

export default {
  chatInference,
  chatInferenceStreaming,
  deleteChatSession,
  createChatRequest,
}
