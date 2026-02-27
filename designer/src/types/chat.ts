/**
 * Chat API Types - aligned with server/api/routers/inference/models.py
 *
 * This file contains types for backend communication and external chat service integration.
 * These types should remain stable and aligned with the API contract.
 */

/**
 * Base message interface containing shared properties between API and UI layers
 */
export interface BaseMessage {
  content: string
}

/**
 * Chat message structure for API communication
 * Used in requests/responses to the chat inference service
 *
 * @example
 * const message: ChatMessage = {
 *   role: 'user',
 *   content: 'Hello, how are you?'
 * }
 */
export interface ChatMessage extends BaseMessage {
  /** Message role for API - must match backend expectations */
  role: 'system' | 'user' | 'assistant' | 'tool'
  /** Tool call ID for tool messages */
  tool_call_id?: string
}

/**
 * Complete chat request payload for API calls
 * Contains messages and all model configuration parameters
 */
export interface ChatRequest {
  model?: string | null
  messages: ChatMessage[]
  metadata?: Record<string, string>
  modalities?: string[]
  response_format?: Record<string, string>
  stream?: boolean | null
  temperature?: number | null
  top_p?: number | null
  top_k?: number | null
  max_tokens?: number | null
  stop?: string[]
  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number>
  // LlamaFarm-specific RAG parameters
  rag_enabled?: boolean | null
  database?: string | null
  rag_retrieval_strategy?: string | null
  rag_top_k?: number | null
  rag_score_threshold?: number | null
  // Thinking/reasoning model parameters (universal runtime)
  think?: boolean | null
  thinking_budget?: number | null
  // RAG sources visibility for debugging/testing
  include_sources?: boolean | null
  sources_limit?: number | null
}

/**
 * Individual response choice from chat API
 */
export interface ChatChoice {
  index: number
  message: ChatMessage
  finish_reason: string
}

/**
 * Token usage statistics from API response
 */
export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

/**
 * Complete chat response from API
 * Contains generated choices and usage metadata
 */
export interface ChatResponse {
  id: string
  object: string
  created: number
  model?: string | null
  choices: ChatChoice[]
  usage?: Usage | null
}

/**
 * Response from session deletion API endpoint
 */
export interface DeleteSessionResponse {
  message: string
}

// Custom error types for better error handling
export class ChatApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: any
  ) {
    super(message)
    this.name = 'ChatApiError'
  }
}

export class NetworkError extends Error {
  constructor(
    message: string,
    public originalError: Error
  ) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public validationErrors: any
  ) {
    super(message)
    this.name = 'ValidationError'
  }
}

// Streaming-specific types

/**
 * Source item returned by RAG retrieval
 */
export interface RAGSource {
  content: string
  source: string
  score: number
  metadata?: Record<string, any>
}

/**
 * Custom SSE event for RAG sources
 */
export interface SourcesEvent {
  type: 'sources'
  sources: RAGSource[]
}

/**
 * OpenAI-compatible chat completion chunk
 */
export interface ChatCompletionChunkEvent {
  id: string
  object: string
  created: number
  model?: string | null
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string
      tool_calls?: Array<{
        index?: number
        id?: string
        type?: string
        function?: {
          name?: string
          arguments?: string
        }
      }>
    }
    finish_reason: string | null
  }>
}

/**
 * Discriminated union of all SSE event types
 */
export type ChatStreamEvent = SourcesEvent | ChatCompletionChunkEvent

/**
 * Type guard for sources event
 */
export function isSourcesEvent(event: ChatStreamEvent): event is SourcesEvent {
  return 'type' in event && event.type === 'sources'
}

/**
 * Type guard for chat completion chunk
 */
export function isChatChunk(
  event: ChatStreamEvent
): event is ChatCompletionChunkEvent {
  return 'choices' in event
}

// Keep ChatStreamChunk as alias for backward compatibility
export type ChatStreamChunk = ChatStreamEvent

/**
 * Callback function type for handling streaming chunks
 */
export type StreamChunkHandler = (chunk: ChatStreamEvent) => void

/**
 * Callback function type for handling streaming errors
 */
export type StreamErrorHandler = (error: Error) => void

/**
 * Callback function type for handling stream completion
 */
export type StreamCompleteHandler = () => void

/**
 * Configuration for streaming chat requests
 */
export interface StreamingChatOptions {
  onChunk?: StreamChunkHandler
  onError?: StreamErrorHandler
  onComplete?: StreamCompleteHandler
  signal?: AbortSignal
}

/**
 * Health response interface (imported conceptually from healthService to avoid circular deps)
 */
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy'
  summary: string
  components: Array<{
    name: string
    status: 'healthy' | 'degraded' | 'unhealthy'
    message: string
    latency_ms?: number
    details?: Record<string, any>
  }>
  seeds: Array<{
    name: string
    status: 'healthy' | 'degraded' | 'unhealthy'
    message: string
    latency_ms?: number
    runtime?: {
      provider: string
      model: string
      host?: string
    }
  }>
  timestamp: number
}

/**
 * Classified error with recovery information
 * Used to provide better error messages and actionable recovery steps
 */
export interface ClassifiedError {
  type: 'server_down' | 'degraded' | 'timeout' | 'validation' | 'unknown'
  title: string
  message: string
  originalError: Error
  healthStatus?: HealthResponse
  recoveryCommands?: Array<{
    description: string
    command: string
  }>
}
