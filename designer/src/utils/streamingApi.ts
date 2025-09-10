/**
 * Streaming API utilities for Server-Sent Events (SSE) chat implementation
 * Handles streaming responses from the chat API with proper error handling and cancellation
 */

import {
  ChatRequest,
  StreamingChatResponse,
  ChatSessionContext,
  StreamControl,
  ChatApiError,
  NetworkError,
} from '../types/chat'

/**
 * Parse server error response and return user-friendly message
 * @param response - HTTP response object
 * @param defaultMessage - Default message if parsing fails
 * @returns User-friendly error message
 */
async function parseServerError(response: Response, defaultMessage: string): Promise<string> {
  try {
    const errorBody = await response.text()
    const errorData = JSON.parse(errorBody)
    
    // Try different error message formats
    return errorData.detail || errorData.message || errorData.error || defaultMessage
  } catch {
    // If JSON parsing fails, return default message
    return defaultMessage
  }
}

/**
 * Create user-friendly error message from error object
 * @param error - Error object
 * @returns User-friendly error message
 */
export function createUserFriendlyErrorMessage(error: unknown): string {
  if (error instanceof ChatApiError) {
    switch (error.status) {
      case 400:
        return 'Invalid request. Please check your message and try again.'
      case 401:
        return 'Authentication failed. Please refresh the page and try again.'
      case 403:
        return 'Access denied. You may not have permission for this project.'
      case 404:
        return 'Project or chat endpoint not found. Please check your project settings.'
      case 429:
        return 'Rate limit exceeded. Please wait a moment and try again.'
      case 500:
        return 'Server error. Please try again in a moment.'
      case 503:
        return 'Chat service temporarily unavailable. Please try again later.'
      default:
        return error.message || 'An unexpected server error occurred'
    }
  }
  
  if (error instanceof NetworkError) {
    if (error.message.includes('cancelled') || error.message.includes('aborted')) {
      return 'Request was cancelled'
    }
    return 'Network connection failed. Please check your internet connection and try again.'
  }
  
  if (error instanceof Error) {
    if (error.name === 'AbortError') {
      return 'Request was cancelled'
    }
    if (error.message.includes('fetch')) {
      return 'Unable to connect to the chat service. Please try again.'
    }
    return error.message
  }
  
  return 'An unexpected error occurred. Please try again.'
}

/**
 * Parse SSE data line and extract JSON payload
 * @param line - Raw SSE line from stream
 * @returns Parsed JSON object or null if invalid
 */
function parseSSELine(line: string): StreamingChatResponse | null {
  const trimmed = line.trim()
  
  // Skip empty lines
  if (!trimmed) return null
  
  // Must start with "data:"
  if (!trimmed.startsWith('data:')) return null
  
  // Extract payload after "data:" prefix
  const payload = trimmed.substring(5).trim()
  
  // Check for stream termination
  if (payload === '[DONE]') return null
  
  try {
    return JSON.parse(payload) as StreamingChatResponse
  } catch (error) {
    console.warn('Failed to parse SSE JSON payload:', payload, error)
    return null
  }
}

/**
 * Build project-specific chat API URL
 * @param context - Chat session context with project info
 * @returns Full API URL for chat completions
 */
export function buildChatAPIURL(context: ChatSessionContext): string {
  const base = context.serverURL.replace(/\/$/, '')
  return `${base}/v1/projects/${encodeURIComponent(context.namespace)}/${encodeURIComponent(context.projectID)}/chat/completions`
  // return `${base}/v1/projects/llamafarm/project-seed/chat/completions`
  
}

/**
 * Build project-specific session deletion URL
 * @param context - Chat session context with project and session info
 * @returns Full API URL for session deletion
 */
export function buildSessionDeleteURL(context: ChatSessionContext): string {
  const base = context.serverURL.replace(/\/$/, '')
  return `${base}/v1/projects/${encodeURIComponent(context.namespace)}/${encodeURIComponent(context.projectID)}/chat/session/${encodeURIComponent(context.sessionID || '')}`
}

/**
 * Start a streaming chat request with proper SSE handling
 * @param request - Chat request payload
 * @param context - Session context with connection info
 * @returns Promise resolving to stream control and chunk/error channels
 */
export async function startChatStream(
  request: ChatRequest,
  context: ChatSessionContext
): Promise<{
  chunks: ReadableStream<string>
  control: StreamControl
  getSessionId: () => string | undefined
}> {
  const abortController = new AbortController()
  let sessionId: string | undefined = context.sessionID
  
  // Build request
  const url = buildChatAPIURL(context)
  const requestPayload: ChatRequest = {
    ...request,
    stream: true
  }
  
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
  }
  
  if (context.sessionID) {
    headers['X-Session-ID'] = context.sessionID
  }
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestPayload),
      signal: abortController.signal,
    })
    
    if (!response.ok) {
      const errorMessage = await parseServerError(response, `Server returned ${response.status}`)
      throw new ChatApiError(errorMessage, response.status)
    }
    
    // Extract session ID from response headers
    const responseSessionId = response.headers.get('X-Session-ID')
    if (responseSessionId) {
      sessionId = responseSessionId
    }
    
    if (!response.body) {
      throw new NetworkError('No response body received', new Error('Empty response body'))
    }
    
    // Create readable stream for chunks
    const chunks = new ReadableStream<string>({
      async start(controller) {
        const reader = response.body!.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        
        try {
          while (true) {
            const { done, value } = await reader.read()
            
            if (done) break
            
            // Decode chunk and add to buffer
            buffer += decoder.decode(value, { stream: true })
            
            // Process complete lines
            const lines = buffer.split('\n')
            buffer = lines.pop() || '' // Keep incomplete line in buffer
            
            for (const line of lines) {
              const data = parseSSELine(line)
              if (data && data.choices && data.choices.length > 0) {
                const delta = data.choices[0].delta
                if (delta.content) {
                  controller.enqueue(delta.content)
                }
              }
            }
          }
        } catch (error) {
          if (error instanceof Error && error.name === 'AbortError') {
            // Stream was cancelled, close cleanly
            controller.close()
          } else {
            controller.error(error)
          }
        } finally {
          reader.releaseLock()
          controller.close()
        }
      },
      
      cancel() {
        abortController.abort()
      }
    })
    
    const control: StreamControl = {
      abort: () => abortController.abort(),
      isActive: !abortController.signal.aborted
    }
    
    return {
      chunks,
      control,
      getSessionId: () => sessionId
    }
    
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new NetworkError('Request was cancelled', error)
    } else if (error instanceof ChatApiError) {
      throw error
    } else {
      throw new NetworkError('Failed to start chat stream', error as Error)
    }
  }
}

/**
 * Send a complete chat request and return the full response as a string
 * @param request - Chat request payload
 * @param context - Session context with connection info
 * @param retryCount - Number of retries attempted (for internal use)
 * @returns Promise resolving to complete response text and session ID
 */
export async function sendChatRequest(
  request: ChatRequest,
  context: ChatSessionContext,
  retryCount = 0
): Promise<{ response: string; sessionId?: string }> {
  const maxRetries = 2
  
  try {
    const { chunks, control, getSessionId } = await startChatStream(request, context)
    
    try {
      const reader = chunks.getReader()
      let fullResponse = ''
      
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        
        fullResponse += value
      }
      
      return {
        response: fullResponse,
        sessionId: getSessionId()
      }
    } catch (error) {
      control.abort()
      throw error
    }
  } catch (error) {
    // Retry on network errors, but not on client errors
    if (retryCount < maxRetries && error instanceof NetworkError) {
      console.warn(`Chat request failed (attempt ${retryCount + 1}/${maxRetries + 1}):`, error.message)
      
      // Exponential backoff delay
      const delay = Math.min(1000 * Math.pow(2, retryCount), 5000)
      await new Promise(resolve => setTimeout(resolve, delay))
      
      return sendChatRequest(request, context, retryCount + 1)
    }
    
    throw error
  }
}

/**
 * Delete a chat session
 * @param context - Session context with project and session info
 * @returns Promise resolving when session is deleted
 */
export async function deleteChatSession(context: ChatSessionContext): Promise<void> {
  if (!context.sessionID) {
    return // No session to delete
  }
  
  const url = buildSessionDeleteURL(context)
  
  try {
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    })
    
    if (!response.ok) {
      // Log error but don't throw - session deletion is not critical
      console.warn(`Failed to delete session ${context.sessionID}: ${response.status}`)
    }
  } catch (error) {
    // Log error but don't throw - session deletion is not critical
    console.warn(`Failed to delete session ${context.sessionID}:`, error)
  }
}
