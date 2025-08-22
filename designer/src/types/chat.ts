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
  role: 'system' | 'user' | 'assistant'
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
  constructor(message: string, public originalError: Error) {
    super(message)
    this.name = 'NetworkError'
  }
}

export class ValidationError extends Error {
  constructor(message: string, public validationErrors: any) {
    super(message)
    this.name = 'ValidationError'
  }
}

