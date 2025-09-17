import { apiClient } from './client'
import {
  ChatRequest,
  ChatResponse,
  DeleteSessionResponse,
} from '../types/chat'


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

  const response = await apiClient.post<ChatResponse>('/inference/chat', chatRequest, {
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
  const response = await apiClient.delete<DeleteSessionResponse>(`/inference/chat/session/${encodeURIComponent(sessionId)}`)
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
