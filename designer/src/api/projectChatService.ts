import { apiClient } from './client'
import { NetworkError, ChatApiError, ValidationError } from '../types/chat'
import { handleSSEResponse } from '../utils/sseUtils'

/**
 * Project Chat API Types - separate from existing chat system
 * These types are specifically for project-based chat conversations
 */

/**
 * LlamaFarm configuration structure based on backend datamodel
 */
export interface LlamaFarmConfig {
  version: 'v1'
  name: string
  namespace: string
  prompts?: Array<{
    name: string
    messages: Array<{
      role?: string
      content: string
    }>
  }>
  rag?: Record<string, any>
  datasets?: Array<{
    name: string
    data_processing_strategy?: string
    database?: string
    files?: string[]
    details?: {
      files_metadata?: Array<{
        original_file_name?: string
        resolved_file_name?: string
        hash?: string
      }>
    }
  }>
  runtime: Record<string, any>
}

/**
 * Project structure from backend API
 */
export interface Project {
  namespace: string
  name: string
  config: LlamaFarmConfig
}

/**
 * Project chat message structure
 */
export interface ProjectChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

/**
 * Project chat request payload
 */
export interface ProjectChatRequest {
  messages: ProjectChatMessage[]
  stream?: boolean
  model?: string | null
  temperature?: number | null
  top_p?: number | null
  top_k?: number | null
  max_tokens?: number | null
  stop?: string[]
  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number>
  // Optional RAG parameters (used by Designer Test Chat)
  rag_enabled?: boolean
  database?: string
  rag_top_k?: number
  rag_score_threshold?: number
}

/**
 * Project chat choice from API response
 */
export interface ProjectChatChoice {
  index: number
  message: ProjectChatMessage
  finish_reason: string
}

/**
 * Token usage statistics
 */
export interface ProjectChatUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

/**
 * Complete project chat response from API
 */
export interface ProjectChatCompletion {
  id: string
  object: string
  created: number
  model?: string | null
  choices: ProjectChatChoice[]
  usage?: ProjectChatUsage | null
}

/**
 * Streaming chunk structure for project chat
 */
export interface ProjectChatStreamChunk {
  id: string
  object: string
  created: number
  model?: string | null
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string
    }
    finish_reason: string | null
  }>
}

/**
 * Callback function types for streaming
 */
export type ProjectChatStreamChunkHandler = (
  chunk: ProjectChatStreamChunk
) => void
export type ProjectChatStreamErrorHandler = (error: Error) => void
export type ProjectChatStreamCompleteHandler = () => void

/**
 * Configuration for streaming project chat requests
 */
export interface ProjectChatStreamingOptions {
  onChunk?: ProjectChatStreamChunkHandler
  onError?: ProjectChatStreamErrorHandler
  onComplete?: ProjectChatStreamCompleteHandler
  signal?: AbortSignal
}

/**
 * Result from project chat API calls including server-provided session ID
 */
export interface ProjectChatResult {
  completion: ProjectChatCompletion
  sessionId: string
}

/**
 * Send a non-streaming project chat message
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param request - The chat request payload
 * @param sessionId - Optional session ID for conversation continuity (if not provided, server creates new session)
 * @returns Promise<ProjectChatResult> - Completion and server-provided session ID
 */
export async function sendProjectChatMessage(
  namespace: string,
  projectId: string,
  request: ProjectChatRequest,
  sessionId?: string
): Promise<ProjectChatResult> {
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

  // Only add session header if we have a session ID (for continuing conversation)
  if (sessionId) {
    headers['X-Session-ID'] = sessionId
  }

  // Ensure stream is false for non-streaming requests
  const chatRequest = { ...request, stream: false }

  try {
    const response = await apiClient.post<ProjectChatCompletion>(
      `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}/chat/completions`,
      chatRequest,
      { headers }
    )

    // Extract session ID from response headers (server provides this)
    const responseSessionId =
      response.headers['x-session-id'] ||
      response.headers['X-Session-ID'] ||
      sessionId || // Fallback to sent session ID
      ''

    if (!responseSessionId) {
      console.warn('No session ID received from server response')
    }

    return {
      completion: response.data,
      sessionId: responseSessionId,
    }
  } catch (error) {
    if (
      error instanceof ChatApiError ||
      error instanceof ValidationError ||
      error instanceof NetworkError
    ) {
      throw error
    }

    throw new NetworkError(
      `Failed to send project chat message: ${error instanceof Error ? error.message : 'Unknown error'}`,
      error instanceof Error ? error : new Error('Unknown error')
    )
  }
}

/**
 * Send a streaming project chat message using Server-Sent Events
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param request - The chat request payload with stream: true
 * @param sessionId - Optional session ID for conversation continuity
 * @param options - Streaming options with callbacks
 * @returns Promise<string> - The final session ID
 */
export async function streamProjectChatMessage(
  namespace: string,
  projectId: string,
  request: ProjectChatRequest,
  sessionId?: string,
  options: ProjectChatStreamingOptions = {}
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

  try {
    // Ensure streaming is enabled
    const streamingRequest = { ...request, stream: true }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
      'Cache-Control': 'no-cache',
    }

    // Only add session header if we have a session ID (for continuing conversation)
    if (sessionId) {
      headers['X-Session-ID'] = sessionId
    }

    // Use fetch directly for streaming instead of axios
    const baseURL = apiClient.defaults.baseURL || ''
    const url = `${baseURL}/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}/chat/completions`

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
      throw new NetworkError(
        `HTTP ${response.status}: ${response.statusText} - ${errorText}`,
        new Error(`Fetch failed with status ${response.status}`)
      )
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

    // Handle the streaming response using SSE utility
    await handleSSEResponse<ProjectChatStreamChunk>(
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
        'Project chat streaming request was cancelled',
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
              : 'Unknown project chat streaming error',
            error instanceof Error ? error : new Error('Unknown error')
          )

    onError?.(networkError)
    throw networkError
  }
}

/**
 * Start a new project chat session (no session ID sent to server)
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param message - The initial message to start the conversation
 * @param options - Additional request options
 * @returns Promise<ProjectChatResult> - The response with server-generated session ID
 */
export async function startNewProjectChatSession(
  namespace: string,
  projectId: string,
  message: string,
  options: Partial<ProjectChatRequest> = {}
): Promise<ProjectChatResult> {
  const request = createProjectChatRequest(message, options)
  // Don't pass sessionId - server will create new session
  return await sendProjectChatMessage(namespace, projectId, request)
}

/**
 * Continue an existing project chat session
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param sessionId - The existing session ID from server
 * @param message - The message to send
 * @param options - Additional request options
 * @returns Promise<ProjectChatResult> - The response with same session ID
 */
export async function continueProjectChatSession(
  namespace: string,
  projectId: string,
  sessionId: string,
  message: string,
  options: Partial<ProjectChatRequest> = {}
): Promise<ProjectChatResult> {
  const request = createProjectChatRequest(message, options)
  return await sendProjectChatMessage(namespace, projectId, request, sessionId)
}

/**
 * Start a new streaming project chat session
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param message - The initial message to start the conversation
 * @param streamingOptions - Streaming callback options
 * @param requestOptions - Additional request options
 * @returns Promise<string> - The server-generated session ID
 */
export async function startNewProjectChatStreamingSession(
  namespace: string,
  projectId: string,
  message: string,
  streamingOptions: ProjectChatStreamingOptions = {},
  requestOptions: Partial<ProjectChatRequest> = {}
): Promise<string> {
  const request = createProjectChatRequest(message, requestOptions)
  // Don't pass sessionId - server will create new session
  return await streamProjectChatMessage(
    namespace,
    projectId,
    request,
    undefined,
    streamingOptions
  )
}

/**
 * Continue an existing streaming project chat session
 * @param namespace - The project namespace
 * @param projectId - The project identifier
 * @param sessionId - The existing session ID from server
 * @param message - The message to send
 * @param streamingOptions - Streaming callback options
 * @param requestOptions - Additional request options
 * @returns Promise<string> - The same session ID
 */
export async function continueProjectChatStreamingSession(
  namespace: string,
  projectId: string,
  sessionId: string,
  message: string,
  streamingOptions: ProjectChatStreamingOptions = {},
  requestOptions: Partial<ProjectChatRequest> = {}
): Promise<string> {
  const request = createProjectChatRequest(message, requestOptions)
  return await streamProjectChatMessage(
    namespace,
    projectId,
    request,
    sessionId,
    streamingOptions
  )
}

/**
 * Helper function to create a simple project chat request
 * @param message - The user message content
 * @param options - Additional request options
 * @returns ProjectChatRequest - The formatted request
 */
export function createProjectChatRequest(
  message: string,
  options: Partial<ProjectChatRequest> = {}
): ProjectChatRequest {
  // Validate message
  if (typeof message !== 'string' || !message.trim()) {
    throw new ValidationError('Message must be a non-empty string', { message })
  }

  return {
    messages: [{ role: 'user', content: message.trim() }],
    ...options,
  }
}

/**
 * Helper function to extract the latest user message from a conversation
 * @param messages - Array of chat messages
 * @returns string | null - The latest user message content or null if not found
 */
export function extractLatestUserMessage(
  messages: ProjectChatMessage[]
): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (message.role === 'user' && message.content) {
      return message.content
    }
  }
  return null
}

/**
 * Helper function to build a conversation history from messages
 * @param messages - Array of chat messages
 * @param maxMessages - Maximum number of messages to include (default: 50)
 * @returns ProjectChatMessage[] - Trimmed conversation history
 */
export function buildConversationHistory(
  messages: ProjectChatMessage[],
  maxMessages: number = 50
): ProjectChatMessage[] {
  if (messages.length <= maxMessages) {
    return [...messages]
  }

  // Keep the most recent messages
  return messages.slice(-maxMessages)
}

/**
 * Default export with all project chat service functions
 */
export default {
  sendProjectChatMessage,
  streamProjectChatMessage,
  startNewProjectChatSession,
  continueProjectChatSession,
  startNewProjectChatStreamingSession,
  continueProjectChatStreamingSession,
  createProjectChatRequest,
  extractLatestUserMessage,
  buildConversationHistory,
}
