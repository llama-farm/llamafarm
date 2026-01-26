import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios'
import { ChatApiError, NetworkError, ValidationError } from '../types/chat'
import { devToolsEmitter } from '../utils/devToolsEmitter'
import type { CapturedRequest } from '../contexts/DevToolsContext'

// Extend axios config to include our DevTools tracking ID
declare module 'axios' {
  interface InternalAxiosRequestConfig {
    _devToolsId?: string
    _devToolsTimestamp?: number
  }
}

// Prefer explicit API host via env; fall back to Vite proxy '/api'
const API_VERSION = import.meta.env.VITE_API_VERSION || 'v1'
const API_HOST = (import.meta.env as any).VITE_APP_API_URL as string | undefined

// Universal Runtime URL for direct TTS/STT calls
// Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
const RUNTIME_BASE_URL = import.meta.env.VITE_UNIVERSAL_RUNTIME_URL || 'http://127.0.0.1:11540'

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
      // Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
      base = `http://127.0.0.1:8000/${API_VERSION}`
    }
    return base
  }

  // 2) Local dev convenience: if running on localhost, prefer direct API to avoid proxy flakiness
  if (
    typeof window !== 'undefined' &&
    window.location.hostname === 'localhost'
  ) {
    // Use 127.0.0.1 instead of localhost to avoid IPv6 resolution issues on macOS
    return `http://127.0.0.1:8000/${API_VERSION}`
  }

  // 3) Default to vite proxy
  return `/api/${API_VERSION}`
}

const API_BASE_URL = resolveBaseUrl()

/**
 * Format validation errors for display
 */
function formatValidationError(errorData: any): string {
  if (!errorData?.detail) return 'Invalid request'

  // Pydantic validation errors are usually arrays
  if (Array.isArray(errorData.detail)) {
    return errorData.detail
      .map((err: any) => {
        const loc = err.loc?.join('.') || 'unknown'
        const msg = err.msg || err.message || 'validation error'
        return `${loc}: ${msg}`
      })
      .join(', ')
  }

  // If detail is a string, return it
  if (typeof errorData.detail === 'string') {
    return errorData.detail
  }

  // Otherwise try to stringify
  return JSON.stringify(errorData.detail)
}

/**
 * Generate a unique ID for DevTools request tracking
 */
function generateDevToolsId(): string {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/**
 * Extract headers from axios config as a plain Record
 */
function extractRequestHeaders(
  config: InternalAxiosRequestConfig
): Record<string, string> {
  const headers: Record<string, string> = {}
  if (config.headers) {
    // AxiosHeaders can be iterated or accessed via toJSON
    const headerObj =
      typeof config.headers.toJSON === 'function'
        ? config.headers.toJSON()
        : config.headers
    for (const [key, value] of Object.entries(headerObj)) {
      if (typeof value === 'string') {
        headers[key] = value
      } else if (value !== undefined && value !== null) {
        headers[key] = String(value)
      }
    }
  }
  return headers
}

/**
 * Get the request body, handling FormData specially
 */
function extractRequestBody(config: InternalAxiosRequestConfig): unknown {
  if (!config.data) return null

  // FormData can't be serialized - return a placeholder
  if (config.data instanceof FormData) {
    const fields: Record<string, string> = {}
    config.data.forEach((value, key) => {
      if (value instanceof File) {
        fields[key] = `[File: ${value.name}, ${value.size} bytes]`
      } else {
        fields[key] = String(value)
      }
    })
    return { _formData: true, fields }
  }

  return config.data
}

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

/**
 * Universal Runtime client for direct TTS/STT calls
 * Uses the same DevTools interceptors as apiClient
 */
export const runtimeClient: AxiosInstance = axios.create({
  baseURL: RUNTIME_BASE_URL,
  timeout: 60000,
})

// =============================================================================
// DevTools Interceptor Setup - Reusable for multiple axios instances
// =============================================================================

/**
 * Setup DevTools request/response interceptors for an axios instance
 */
function setupDevToolsInterceptors(client: AxiosInstance): void {
  // Request interceptor - Captures outgoing requests
  client.interceptors.request.use(
    config => {
      // Skip DevTools capture for health check endpoints (they poll frequently)
      const requestUrl = config.url || ''
      if (requestUrl === '/health' || requestUrl.endsWith('/health')) {
        return config
      }

      // Only capture if DevTools has subscribers (context is mounted)
      if (!devToolsEmitter.hasSubscribers()) {
        return config
      }

      // Generate tracking ID and timestamp
      const id = generateDevToolsId()
      config._devToolsId = id
      config._devToolsTimestamp = Date.now()

      // Build full URL
      const baseURL = config.baseURL || ''
      const url = config.url || ''
      const fullUrl = url.startsWith('http') ? url : `${baseURL}${url}`

      // Determine HTTP method
      const method = (config.method?.toUpperCase() || 'GET') as CapturedRequest['method']

      // Emit request event
      devToolsEmitter.emit({
        type: 'request',
        request: {
          id,
          requestId: null,
          method,
          url,
          fullUrl,
          headers: extractRequestHeaders(config),
          body: extractRequestBody(config),
          timestamp: config._devToolsTimestamp,
          isStreaming: false, // Axios requests are not streaming (fetch is used for streaming)
        },
      })

      return config
    },
    error => {
      // Request setup errors are rare but possible
      return Promise.reject(error)
    }
  )

  // Response interceptor - Captures responses
  client.interceptors.response.use(
    response => {
      const config = response.config
      const id = config._devToolsId

      // Only capture if we have a tracking ID
      if (id && devToolsEmitter.hasSubscribers()) {
        // Extract response headers
        const headers: Record<string, string> = {}
        if (response.headers) {
          for (const [key, value] of Object.entries(response.headers)) {
            if (typeof value === 'string') {
              headers[key] = value
            }
          }
        }

        devToolsEmitter.emit({
          type: 'response',
          id,
          response: {
            status: response.status,
            statusText: response.statusText,
            headers,
            body: response.data,
            requestId: headers['x-request-id'] || headers['X-Request-ID'] || null,
          },
        })
      }

      return response
    },
    (error: AxiosError) => {
      // Capture error in DevTools before transforming it
      const config = error.config
      const id = config?._devToolsId

      if (id && devToolsEmitter.hasSubscribers()) {
        // If there's a response, capture it as an error response
        if (error.response) {
          const headers: Record<string, string> = {}
          if (error.response.headers) {
            for (const [key, value] of Object.entries(error.response.headers)) {
              if (typeof value === 'string') {
                headers[key] = value
              }
            }
          }

          devToolsEmitter.emit({
            type: 'response',
            id,
            response: {
              status: error.response.status,
              statusText: error.response.statusText,
              headers,
              body: error.response.data,
              requestId: headers['x-request-id'] || headers['X-Request-ID'] || null,
            },
          })
        } else {
          // Network error or timeout - no response
          devToolsEmitter.emit({
            type: 'error',
            id,
            error: error.message || 'Network error',
          })
        }
      }

      // Pass through to the next error handler
      return Promise.reject(error)
    }
  )
}

// Apply DevTools interceptors to both clients
setupDevToolsInterceptors(apiClient)
setupDevToolsInterceptors(runtimeClient)

// =============================================================================
// Error Transformation Interceptor - Converts errors to typed exceptions
// (Only for apiClient - runtimeClient uses simpler error handling)
// =============================================================================
apiClient.interceptors.response.use(
  response => response,
  (error: AxiosError) => {
    // Silently ignore canceled requests (e.g., from AbortController in React's StrictMode cleanup)
    // These are intentional cancellations, not actual errors
    if (error.code === 'ERR_CANCELED') {
      // Re-throw as a special canceled error that callers can detect
      const canceledError = new Error('Request was canceled')
      ;(canceledError as any).isCanceled = true
      throw canceledError
    }

    if (error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK') {
      throw new NetworkError('Network error occurred', error)
    }

    if (error.response) {
      const { status, data } = error.response
      const errorData = data as any // Type assertion for error response data

      switch (status) {
        case 400:
          throw new ValidationError(
            `Validation error: ${formatValidationError(errorData)}`,
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
            `Validation error: ${formatValidationError(errorData)}`,
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

// Convenience helpers for examples endpoints (typed lightly here)
export const examplesApi = {
  async listAllDatasets() {
    const { data } = await apiClient.get('/examples/datasets')
    return data as { datasets: any[] }
  },
  async listExampleDatasets(exampleId: string) {
    const { data } = await apiClient.get(
      `/examples/${encodeURIComponent(exampleId)}/datasets`
    )
    return data as { datasets: any[] }
  },
  async importExampleDataset(exampleId: string, body: any) {
    const { data } = await apiClient.post(
      `/examples/${encodeURIComponent(exampleId)}/import-dataset`,
      body
    )
    return data as {
      project: string
      namespace: string
      dataset: string
      file_count: number
      task_id?: string
    }
  },
}
