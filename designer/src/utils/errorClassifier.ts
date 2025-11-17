/**
 * Error Classification Utility
 * 
 * Classifies errors from API calls into actionable categories
 * to provide better user feedback and recovery suggestions.
 */

import { NetworkError, ValidationError, ChatApiError, ClassifiedError, HealthResponse } from '../types/chat'

/**
 * Error type categories
 */
export type ErrorType = 
  | 'server_down'      // Server is not accessible at all
  | 'degraded'         // Server responds but services are unhealthy
  | 'timeout'          // Request timed out
  | 'validation'       // Invalid request (400/422)
  | 'unknown'          // Other errors

/**
 * Check if error is a network connectivity error
 */
function isNetworkError(error: Error): boolean {
  // Check for axios/fetch network errors
  if (error instanceof NetworkError) {
    return true
  }

  // Check error message patterns
  const message = error.message.toLowerCase()
  return (
    message.includes('network error') ||
    message.includes('failed to fetch') ||
    message.includes('connection') ||
    message.includes('econnrefused') ||
    message.includes('err_connection') ||
    message.includes('err_network')
  )
}

/**
 * Check if error is a timeout error
 */
function isTimeoutError(error: Error): boolean {
  const message = error.message.toLowerCase()
  return (
    message.includes('timeout') ||
    message.includes('timed out') ||
    message.includes('econnaborted') ||
    error.name === 'TimeoutError'
  )
}

/**
 * Check if error is a validation error
 */
function isValidationError(error: Error): boolean {
  if (error instanceof ValidationError) {
    return true
  }

  if (error instanceof ChatApiError) {
    return error.status === 400 || error.status === 422
  }

  return false
}

/**
 * Classify an error into a specific type with user-friendly messaging
 * 
 * @param error - The error to classify
 * @param healthStatus - Optional health status if already fetched
 * @returns ClassifiedError with type, messages, and recovery info
 */
export function classifyError(
  error: Error,
  healthStatus?: HealthResponse
): ClassifiedError {
  // Check for network/connection errors first
  if (isNetworkError(error)) {
    return {
      type: 'server_down',
      title: 'Server not responding',
      message: 'Unable to connect to the LlamaFarm server. It may not be running.',
      originalError: error,
      healthStatus,
    }
  }

  // Check for timeout errors
  if (isTimeoutError(error)) {
    return {
      type: 'timeout',
      title: 'Request timed out',
      message: 'The server took too long to respond (>60s). It may be overloaded or stuck.',
      originalError: error,
      healthStatus,
    }
  }

  // Check for validation errors
  if (isValidationError(error)) {
    let validationMessage = 'The request was invalid.'
    
    if (error instanceof ValidationError) {
      // Extract validation details if available
      validationMessage = error.message
    } else if (error instanceof ChatApiError && error.response) {
      validationMessage = error.response.detail || error.message
    }

    return {
      type: 'validation',
      title: 'Invalid request',
      message: validationMessage,
      originalError: error,
      healthStatus,
    }
  }

  // Check if we have health status indicating degraded services
  if (healthStatus && healthStatus.status !== 'healthy') {
    return {
      type: 'degraded',
      title: 'Server degraded',
      message: 'The server is running but some services are unavailable.',
      originalError: error,
      healthStatus,
    }
  }

  // Unknown error type
  return {
    type: 'unknown',
    title: 'An error occurred',
    message: error.message || 'An unexpected error occurred.',
    originalError: error,
    healthStatus,
  }
}

/**
 * Get a user-friendly error title based on error type
 */
export function getErrorTitle(type: ErrorType): string {
  switch (type) {
    case 'server_down':
      return 'Server not responding'
    case 'degraded':
      return 'Server degraded'
    case 'timeout':
      return 'Request timed out'
    case 'validation':
      return 'Invalid request'
    case 'unknown':
    default:
      return 'An error occurred'
  }
}

/**
 * Get detailed contextual error message for inline display in chat
 * 
 * @param classified - The classified error
 * @returns Formatted markdown error message with recovery instructions
 */
export function getContextualErrorMessage(classified: ClassifiedError): string {
  switch (classified.type) {
    case 'server_down':
      return `I can't connect to the LlamaFarm server. It appears to be offline.\n\n**To fix this:**\n1. Open a terminal\n2. Run: \`lf start\`\n3. Wait for the server to start\n4. Try your question again`
    
    case 'timeout':
      return `The server is taking too long to respond (timed out after 60s).\n\nThis might mean the server is overloaded or stuck. Try restarting it with \`lf start\`.`
    
    case 'degraded':
      return `The server is running but some services are unavailable.\n\n${classified.message}\n\nCheck the server logs or try restarting with \`lf start\`.`
    
    case 'validation':
      return `There was a problem with the request:\n\n${classified.message}\n\nThis might be a configuration issue. Check your \`llamafarm.yaml\` file.`
    
    default:
      return `I encountered an error: ${classified.message}\n\nPlease try again or check the server status.`
  }
}

/**
 * Determine if health check should be attempted for this error
 */
export function shouldCheckHealth(error: Error): boolean {
  // Only check health for network errors where server might be partially available
  return isNetworkError(error) && !isTimeoutError(error)
}

export default {
  classifyError,
  getErrorTitle,
  getContextualErrorMessage,
  shouldCheckHealth,
}

