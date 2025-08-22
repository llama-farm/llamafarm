// Chat API Types - aligned with server/api/routers/inference/models.py

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

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

export interface ChatChoice {
  index: number
  message: ChatMessage
  finish_reason: string
}

export interface Usage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export interface ChatResponse {
  id: string
  object: string
  created: number
  model?: string | null
  choices: ChatChoice[]
  usage?: Usage | null
}

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

