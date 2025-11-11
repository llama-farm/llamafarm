/**
 * Error Classification Utility
 * 
 * Classifies errors from API calls into actionable categories
 * to provide better user feedback and recovery suggestions.
 */

import { NetworkError, ValidationError, ChatApiError } from '../types/chat'
import { HealthResponse } from '../api/healthService'

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
 * Classified error with recovery information
 */
export interface ClassifiedError {
  type: ErrorType
  title: string
  message: string
  originalError: Error
  healthStatus?: HealthResponse
  shouldCheckHealth: boolean
}

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
      shouldCheckHealth: true, // Try to get health status
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
      shouldCheckHealth: false, // Don't check health on timeout
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
      shouldCheckHealth: false,
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
      shouldCheckHealth: false, // We already have health status
    }
  }

  // Unknown error type
  return {
    type: 'unknown',
    title: 'An error occurred',
    message: error.message || 'An unexpected error occurred.',
    originalError: error,
    healthStatus,
    shouldCheckHealth: false,
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
 * Determine if health check should be attempted for this error
 */
export function shouldCheckHealth(error: Error): boolean {
  // Only check health for network errors where server might be partially available
  return isNetworkError(error) && !isTimeoutError(error)
}

export default {
  classifyError,
  getErrorTitle,
  shouldCheckHealth,
}

