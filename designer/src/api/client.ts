import axios, { AxiosInstance, AxiosError } from 'axios'
import {
  ChatApiError,
  NetworkError,
  ValidationError
} from '../types/chat'

// Use '/api' path consistently - Vite proxy will handle routing
const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1'
const API_BASE_URL = `/api/${API_VERSION}`

/**
 * Shared API client instance with common configuration
 * Can be imported and used by all service modules
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // Timeout for API operations (30 seconds)
})

// Response interceptor for consistent error handling across all services
apiClient.interceptors.response.use(
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

// Export the configured client as default as well for convenience
export default apiClient
