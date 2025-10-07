import axios, { AxiosInstance, AxiosError } from 'axios'
import { ChatApiError, NetworkError, ValidationError } from '../types/chat'

// Prefer explicit API host via env; fall back to Vite proxy '/api'
const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1'
const API_HOST = (import.meta.env as any).VITE_APP_API_URL as string | undefined

function resolveBaseUrl(): string {
  // 1) Explicit host from env
  if (API_HOST && typeof API_HOST === 'string' && API_HOST.trim().length > 0) {
    let base = `${API_HOST.replace(/\/$/, '')}/${API_VERSION}`
    // If env points at docker hostname `server`, but we're on localhost, fall back
    if (
      base.includes('://server:') &&
      typeof window !== 'undefined' &&
      window.location.hostname === 'localhost'
    ) {
      base = `http://localhost:8000/${API_VERSION}`
    }
    return base
  }

  // 2) Local dev convenience: if running on localhost, prefer direct API to avoid proxy flakiness
  if (
    typeof window !== 'undefined' &&
    window.location.hostname === 'localhost'
  ) {
    return `http://localhost:8000/${API_VERSION}`
  }

  // 3) Default to vite proxy
  return `/api/${API_VERSION}`
}

const API_BASE_URL = resolveBaseUrl()

/**
 * Shared API client instance with common configuration
 * Can be imported and used by all service modules
 */
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // Timeout for API operations (60 seconds)
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
