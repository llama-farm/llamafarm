import axios, { AxiosInstance, AxiosError } from 'axios'
import {
  ChatRequest,
  ChatResponse,
  DeleteSessionResponse,
  ChatApiError,
  NetworkError,
  ValidationError
} from '../types/chat'

// Use '/api' path consistently - Vite proxy will handle routing
const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1'
const CHAT_API_BASE_URL = `/api/${API_VERSION}/inference`

// Axios instance for chat API
const chatHttp: AxiosInstance = axios.create({
  baseURL: CHAT_API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // Timeout for chat operations (30 seconds)
})

// Response interceptor for consistent error handling
chatHttp.interceptors.response.use(
  response => response,
  (error: AxiosError) => {
    if (error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK') {
      throw new NetworkError('Network error occurred', error)
    }

    if (error.response) {
      const { status, data } = error.response
      const errorData = data as any // Type assertion for error response data
      
      switch (status) {
        case 400:
          throw new ValidationError(
            `Validation error: ${errorData?.detail || 'Invalid request'}`,
            errorData
          )
        case 404:
          throw new ChatApiError(
            `Resource not found: ${errorData?.detail || 'Not found'}`,
            status,
            errorData
          )
        case 422:
          throw new ValidationError(
            `Validation error: ${errorData?.detail || 'Unprocessable entity'}`,
            errorData
          )
        case 500:
          throw new ChatApiError(
            `Server error: ${errorData?.detail || 'Internal server error'}`,
            status,
            errorData
          )
        default:
          throw new ChatApiError(
            `HTTP ${status}: ${errorData?.detail || error.message}`,
            status,
            errorData
          )
      }
    }

    throw new NetworkError('Unknown error occurred', error)
  }
)


/**
 * Send a chat message to the inference endpoint
 * @param chatRequest - The chat request payload
 * @param sessionId - Optional session ID for conversation continuity
 * @returns Promise<{data: ChatResponse, sessionId: string}>
 */
export async function chatInference(
  chatRequest: ChatRequest,
  sessionId?: string
): Promise<{data: ChatResponse, sessionId: string}> {
  const headers: Record<string, string> = {}
  
  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  const response = await chatHttp.post<ChatResponse>('/chat', chatRequest, {
    headers,
  })

  // Extract session ID from response headers (server provides this)
  const responseSessionId = response.headers['x-session-id'] || sessionId || ''

  return {
    data: response.data,
    sessionId: responseSessionId
  }
}

/**
 * Delete a chat session
 * @param sessionId - The session ID to delete
 * @returns Promise<DeleteSessionResponse>
 */
export async function deleteChatSession(sessionId: string): Promise<DeleteSessionResponse> {
  const response = await chatHttp.delete<DeleteSessionResponse>(`/chat/session/${encodeURIComponent(sessionId)}`)
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
  return {
    messages: [{ role: 'user', content: message }],
    metadata: {},
    modalities: [],
    response_format: {},
    stop: [],
    logit_bias: {},
    ...options,
  }
}

export default {
  chatInference,
  deleteChatSession,
  createChatRequest,
}
