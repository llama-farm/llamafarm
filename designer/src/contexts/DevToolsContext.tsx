import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  type ReactNode,
} from 'react'
import { devToolsEmitter } from '../utils/devToolsEmitter'

export interface CapturedRequest {
  id: string // Client-generated UUID
  requestId: string | null // X-Request-ID from server
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  url: string // Path only (e.g., /v1/chat/completions)
  fullUrl: string // Full URL with host
  headers: Record<string, string>
  body: any
  timestamp: number // Start time
  duration?: number // Calculated on completion
  status?: number
  statusText?: string
  responseHeaders?: Record<string, string>
  responseBody?: any
  isStreaming: boolean
  streamChunks: any[]
  streamComplete: boolean
  error?: string
}

type ActiveTab = 'request' | 'response' | 'code'

interface DevToolsContextValue {
  // State
  requests: CapturedRequest[]
  selectedRequest: CapturedRequest | null
  isExpanded: boolean
  activeTab: ActiveTab

  // Actions
  captureRequest: (req: Omit<CapturedRequest, 'streamChunks' | 'streamComplete'>) => void
  updateResponse: (
    id: string,
    response: {
      status: number
      statusText: string
      headers: Record<string, string>
      body: any
      requestId?: string | null
    }
  ) => void
  addStreamChunk: (id: string, chunk: any) => void
  completeStream: (id: string) => void
  setError: (id: string, error: string) => void
  selectRequest: (request: CapturedRequest | null) => void
  setIsExpanded: (expanded: boolean) => void
  setActiveTab: (tab: ActiveTab) => void
  clearHistory: () => void
}

const DevToolsContext = createContext<DevToolsContextValue | null>(null)

const MAX_REQUESTS = 50
const STORAGE_KEY = 'lf_devtools_requests'

// Helper to safely parse stored requests
function loadStoredRequests(): CapturedRequest[] {
  if (typeof window === 'undefined') return []
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    if (!stored) return []
    return JSON.parse(stored) as CapturedRequest[]
  } catch {
    return []
  }
}

interface DevToolsProviderProps {
  children: ReactNode
}

export function DevToolsProvider({ children }: DevToolsProviderProps) {
  const [requests, setRequests] = useState<CapturedRequest[]>(loadStoredRequests)
  const [selectedRequest, setSelectedRequest] = useState<CapturedRequest | null>(null)

  // Persist expanded state to localStorage
  const [isExpanded, setIsExpandedState] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false
    try {
      return localStorage.getItem('lf_devtools_expanded') === 'true'
    } catch {
      // localStorage may be disabled (private mode, security settings)
      return false
    }
  })

  const [activeTab, setActiveTab] = useState<ActiveTab>('request')

  // Persist expanded state
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      localStorage.setItem('lf_devtools_expanded', String(isExpanded))
    } catch {
      // Ignore quota or security errors
    }
  }, [isExpanded])

  // Persist requests to sessionStorage
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(requests))
    } catch {
      // Ignore quota errors
    }
  }, [requests])

  const setIsExpanded = useCallback((expanded: boolean) => {
    setIsExpandedState(expanded)
  }, [])

  const captureRequest = useCallback(
    (req: Omit<CapturedRequest, 'streamChunks' | 'streamComplete'>) => {
      const newRequest: CapturedRequest = {
        ...req,
        streamChunks: [],
        streamComplete: false,
      }

      setRequests(prev => {
        const updated = [newRequest, ...prev]
        // Auto-prune to MAX_REQUESTS
        return updated.slice(0, MAX_REQUESTS)
      })

      // Auto-select the new request
      setSelectedRequest(newRequest)
    },
    []
  )

  const updateResponse = useCallback(
    (
      id: string,
      response: {
        status: number
        statusText: string
        headers: Record<string, string>
        body: any
        requestId?: string | null
      }
    ) => {
      setRequests(prev =>
        prev.map(req => {
          if (req.id !== id) return req
          const updated: CapturedRequest = {
            ...req,
            status: response.status,
            statusText: response.statusText,
            responseHeaders: response.headers,
            responseBody: response.body,
            requestId: response.requestId ?? req.requestId,
            duration: Date.now() - req.timestamp,
            // Only mark streamComplete for non-streaming requests
            // Streaming requests are completed via completeStream()
            streamComplete: req.isStreaming ? req.streamComplete : true,
          }
          // Update selected if this is the selected request
          setSelectedRequest(current =>
            current?.id === id ? updated : current
          )
          return updated
        })
      )
    },
    []
  )

  const addStreamChunk = useCallback((id: string, chunk: any) => {
    setRequests(prev =>
      prev.map(req => {
        if (req.id !== id) return req
        const updated: CapturedRequest = {
          ...req,
          streamChunks: [...req.streamChunks, chunk],
        }
        // Update selected if this is the selected request
        setSelectedRequest(current =>
          current?.id === id ? updated : current
        )
        return updated
      })
    )
  }, [])

  const completeStream = useCallback((id: string) => {
    setRequests(prev =>
      prev.map(req => {
        if (req.id !== id) return req
        const updated: CapturedRequest = {
          ...req,
          streamComplete: true,
          duration: Date.now() - req.timestamp,
        }
        // Update selected if this is the selected request
        setSelectedRequest(current =>
          current?.id === id ? updated : current
        )
        return updated
      })
    )
  }, [])

  const setError = useCallback((id: string, error: string) => {
    setRequests(prev =>
      prev.map(req => {
        if (req.id !== id) return req
        const updated: CapturedRequest = {
          ...req,
          error,
          duration: Date.now() - req.timestamp,
          streamComplete: true,
        }
        // Update selected if this is the selected request
        setSelectedRequest(current =>
          current?.id === id ? updated : current
        )
        return updated
      })
    )
  }, [])

  const selectRequest = useCallback((request: CapturedRequest | null) => {
    setSelectedRequest(request)
  }, [])

  const clearHistory = useCallback(() => {
    setRequests([])
    setSelectedRequest(null)
    try {
      sessionStorage.removeItem(STORAGE_KEY)
    } catch {
      // Ignore errors
    }
  }, [])

  // Store callbacks in refs so the subscription doesn't need to be recreated
  const captureRequestRef = useRef(captureRequest)
  const updateResponseRef = useRef(updateResponse)
  const setErrorRef = useRef(setError)

  // Keep refs in sync with latest callbacks
  useEffect(() => {
    captureRequestRef.current = captureRequest
    updateResponseRef.current = updateResponse
    setErrorRef.current = setError
  }, [captureRequest, updateResponse, setError])

  // Subscribe to global DevTools emitter for axios interceptor events
  useEffect(() => {
    const unsubscribe = devToolsEmitter.subscribe(event => {
      switch (event.type) {
        case 'request':
          captureRequestRef.current(event.request)
          break
        case 'response':
          updateResponseRef.current(event.id, event.response)
          break
        case 'error':
          setErrorRef.current(event.id, event.error)
          break
      }
    })

    return unsubscribe
  }, [])

  const value: DevToolsContextValue = {
    requests,
    selectedRequest,
    isExpanded,
    activeTab,
    captureRequest,
    updateResponse,
    addStreamChunk,
    completeStream,
    setError,
    selectRequest,
    setIsExpanded,
    setActiveTab,
    clearHistory,
  }

  return (
    <DevToolsContext.Provider value={value}>
      {children}
    </DevToolsContext.Provider>
  )
}

export function useDevTools() {
  const context = useContext(DevToolsContext)
  if (!context) {
    throw new Error('useDevTools must be used within a DevToolsProvider')
  }
  return context
}

export function useDevToolsOptional() {
  return useContext(DevToolsContext)
}
