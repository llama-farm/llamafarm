import { useState, useCallback, useRef } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import {
  startNewProjectChatSession,
  continueProjectChatSession,
  startNewProjectChatStreamingSession,
  continueProjectChatStreamingSession,
  ProjectChatStreamingOptions,
  ProjectChatRequest,
  ProjectChatResult,
} from '../api/projectChatService'
import { projectChatKeys } from './useProjectChat'

/**
 * Project chat session state interface
 */
export interface ProjectChatSessionState {
  sessionId: string | null
  isSessionActive: boolean
  sessionError: Error | null
}

/**
 * Base session management configuration
 */
interface SessionManagerConfig {
  namespace?: string
  projectId?: string
}

/**
 * Base session management hook that provides common session functionality
 * Used by both regular and streaming session hooks
 */
function useProjectChatSessionBase(config: SessionManagerConfig) {
  const { namespace, projectId } = config
  const queryClient = useQueryClient()
  const [sessionState, setSessionState] = useState<ProjectChatSessionState>({
    sessionId: null,
    isSessionActive: false,
    sessionError: null,
  })

  // Ref to track if we're in the middle of starting a session
  const startingSessionRef = useRef(false)

  /**
   * Common session success handler
   */
  const handleSessionSuccess = useCallback((sessionId: string) => {
    setSessionState({
      sessionId,
      isSessionActive: true,
      sessionError: null,
    })
    
    // Invalidate conversation queries for this project
    if (namespace && projectId) {
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(namespace, projectId) 
      })
    }
    
    startingSessionRef.current = false
  }, [namespace, projectId, queryClient])

  /**
   * Common session error handler
   */
  const handleSessionError = useCallback((error: unknown, operation: string) => {
    const sessionError = error instanceof Error ? error : new Error(`Failed to ${operation}`)
    setSessionState(prev => ({
      ...prev,
      sessionError,
    }))
    startingSessionRef.current = false
    console.error(`Failed to ${operation}:`, error)
  }, [])

  /**
   * Update session state (for continuing sessions)
   */
  const updateSessionState = useCallback((sessionId: string) => {
    setSessionState(prev => ({
      ...prev,
      sessionId,
      sessionError: null,
    }))
    
    // Invalidate conversation queries for this project
    if (namespace && projectId) {
      queryClient.invalidateQueries({ 
        queryKey: projectChatKeys.conversation(namespace, projectId) 
      })
    }
  }, [namespace, projectId, queryClient])

  /**
   * Set an existing session ID (from external source)
   */
  const setSessionId = useCallback((sessionId: string | null) => {
    setSessionState({
      sessionId,
      isSessionActive: !!sessionId,
      sessionError: null,
    })
  }, [])

  /**
   * Clear the current session (start fresh)
   */
  const clearSession = useCallback(() => {
    setSessionState({
      sessionId: null,
      isSessionActive: false,
      sessionError: null,
    })
    startingSessionRef.current = false
  }, [])

  /**
   * Reset session state (useful when switching projects)
   */
  const resetSessionState = useCallback(() => {
    clearSession()
  }, [clearSession])

  return {
    sessionState,
    setSessionState,
    startingSessionRef,
    handleSessionSuccess,
    handleSessionError,
    updateSessionState,
    setSessionId,
    clearSession,
    resetSessionState,
    
    // Computed states
    hasActiveSession: sessionState.isSessionActive && !!sessionState.sessionId,
  }
}

/**
 * Hook to manage project chat sessions with server-provided session IDs
 * Provides session creation, continuation, and cleanup functionality
 */
export const useProjectChatSession = (namespace?: string, projectId?: string) => {
  const base = useProjectChatSessionBase({ namespace, projectId })
  const { sessionState, setSessionState, startingSessionRef, handleSessionSuccess, handleSessionError, updateSessionState } = base

  /**
   * Mutation to start a new session with the server
   */
  const startSessionMutation = useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      message, 
      options 
    }: { 
      namespace: string
      projectId: string
      message: string
      options?: Partial<ProjectChatRequest>
    }): Promise<ProjectChatResult> => {
      return await startNewProjectChatSession(namespace, projectId, message, options)
    },
    onSuccess: (result) => {
      handleSessionSuccess(result.sessionId)
    },
    onError: (error) => {
      handleSessionError(error, 'start session')
    }
  })

  /**
   * Mutation to continue an existing session
   */
  const continueSessionMutation = useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      sessionId,
      message, 
      options 
    }: { 
      namespace: string
      projectId: string
      sessionId: string
      message: string
      options?: Partial<ProjectChatRequest>
    }): Promise<ProjectChatResult> => {
      return await continueProjectChatSession(namespace, projectId, sessionId, message, options)
    },
    onSuccess: (result) => {
      updateSessionState(result.sessionId)
    },
    onError: (error) => {
      handleSessionError(error, 'continue session')
    }
  })

  /**
   * Start a new session with an initial message
   */
  const startNewSession = useCallback(async (
    message: string, 
    options?: Partial<ProjectChatRequest>
  ): Promise<ProjectChatResult | null> => {
    if (!namespace || !projectId) {
      const error = new Error('Namespace and project ID are required to start a session')
      setSessionState(prev => ({ ...prev, sessionError: error }))
      return null
    }

    if (startingSessionRef.current) {
      return null // Already starting a session
    }

    startingSessionRef.current = true
    
    try {
      const result = await startSessionMutation.mutateAsync({ 
        namespace, 
        projectId, 
        message, 
        options 
      })
      return result
    } catch (error) {
      return null
    }
  }, [namespace, projectId, startSessionMutation])

  /**
   * Continue the current session with a message
   */
  const continueSession = useCallback(async (
    message: string, 
    options?: Partial<ProjectChatRequest>
  ): Promise<ProjectChatResult | null> => {
    if (!namespace || !projectId || !sessionState.sessionId) {
      const error = new Error('Active session required to continue conversation')
      setSessionState(prev => ({ ...prev, sessionError: error }))
      return null
    }

    try {
      const result = await continueSessionMutation.mutateAsync({ 
        namespace, 
        projectId, 
        sessionId: sessionState.sessionId,
        message, 
        options 
      })
      return result
    } catch (error) {
      return null
    }
  }, [namespace, projectId, sessionState.sessionId, continueSessionMutation])

  /**
   * Send a message (start new session if none exists, continue if session exists)
   */
  const sendMessage = useCallback(async (
    message: string, 
    options?: Partial<ProjectChatRequest>
  ): Promise<ProjectChatResult | null> => {
    if (sessionState.isSessionActive && sessionState.sessionId) {
      return await continueSession(message, options)
    } else {
      return await startNewSession(message, options)
    }
  }, [sessionState.isSessionActive, sessionState.sessionId, continueSession, startNewSession])

  return {
    // Session state
    sessionId: sessionState.sessionId,
    isSessionActive: sessionState.isSessionActive,
    sessionError: sessionState.sessionError,
    
    // Session operations
    startNewSession,
    continueSession,
    sendMessage,
    ...base, // Include all base operations
    
    // Mutation states
    isStartingSession: startSessionMutation.isPending,
    isContinuingSession: continueSessionMutation.isPending,
    startSessionError: startSessionMutation.error,
    continueSessionError: continueSessionMutation.error,
    
    // Computed states
    hasActiveSession: base.hasActiveSession,
    isLoading: startSessionMutation.isPending || continueSessionMutation.isPending,
    error: sessionState.sessionError || startSessionMutation.error || continueSessionMutation.error,
  }
}

/**
 * Hook for streaming project chat sessions with server-managed session IDs
 */
export const useProjectChatStreamingSession = (namespace?: string, projectId?: string) => {
  const base = useProjectChatSessionBase({ namespace, projectId })
  const { sessionState, setSessionState, startingSessionRef, handleSessionSuccess, handleSessionError, updateSessionState } = base

  /**
   * Mutation to start a new streaming session
   */
  const startStreamingSessionMutation = useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      message, 
      streamingOptions,
      requestOptions 
    }: { 
      namespace: string
      projectId: string
      message: string
      streamingOptions?: ProjectChatStreamingOptions
      requestOptions?: Partial<ProjectChatRequest>
    }): Promise<string> => {
      return await startNewProjectChatStreamingSession(
        namespace, 
        projectId, 
        message, 
        streamingOptions,
        requestOptions
      )
    },
    onSuccess: (sessionId) => {
      handleSessionSuccess(sessionId)
    },
    onError: (error) => {
      handleSessionError(error, 'start streaming session')
    }
  })

  /**
   * Mutation to continue a streaming session
   */
  const continueStreamingSessionMutation = useMutation({
    mutationFn: async ({ 
      namespace, 
      projectId, 
      sessionId,
      message, 
      streamingOptions,
      requestOptions 
    }: { 
      namespace: string
      projectId: string
      sessionId: string
      message: string
      streamingOptions?: ProjectChatStreamingOptions
      requestOptions?: Partial<ProjectChatRequest>
    }): Promise<string> => {
      return await continueProjectChatStreamingSession(
        namespace, 
        projectId, 
        sessionId,
        message, 
        streamingOptions,
        requestOptions
      )
    },
    onSuccess: (sessionId) => {
      updateSessionState(sessionId)
    },
    onError: (error) => {
      handleSessionError(error, 'continue streaming session')
    }
  })

  /**
   * Start a new streaming session with an initial message
   */
  const startNewStreamingSession = useCallback(async (
    message: string, 
    streamingOptions?: ProjectChatStreamingOptions,
    requestOptions?: Partial<ProjectChatRequest>
  ): Promise<string | null> => {
    if (!namespace || !projectId) {
      const error = new Error('Namespace and project ID are required to start a streaming session')
      setSessionState(prev => ({ ...prev, sessionError: error }))
      return null
    }

    if (startingSessionRef.current) {
      return null // Already starting a session
    }

    startingSessionRef.current = true
    
    try {
      const sessionId = await startStreamingSessionMutation.mutateAsync({ 
        namespace, 
        projectId, 
        message, 
        streamingOptions,
        requestOptions 
      })
      return sessionId
    } catch (error) {
      return null
    }
  }, [namespace, projectId, startStreamingSessionMutation])

  /**
   * Continue the current streaming session with a message
   */
  const continueStreamingSession = useCallback(async (
    message: string, 
    streamingOptions?: ProjectChatStreamingOptions,
    requestOptions?: Partial<ProjectChatRequest>
  ): Promise<string | null> => {
    if (!namespace || !projectId || !sessionState.sessionId) {
      const error = new Error('Active session required to continue streaming conversation')
      setSessionState(prev => ({ ...prev, sessionError: error }))
      return null
    }

    try {
      const sessionId = await continueStreamingSessionMutation.mutateAsync({ 
        namespace, 
        projectId, 
        sessionId: sessionState.sessionId,
        message, 
        streamingOptions,
        requestOptions 
      })
      return sessionId
    } catch (error) {
      return null
    }
  }, [namespace, projectId, sessionState.sessionId, continueStreamingSessionMutation])

  /**
   * Send a streaming message (start new session if none exists, continue if session exists)
   */
  const sendStreamingMessage = useCallback(async (
    message: string, 
    streamingOptions?: ProjectChatStreamingOptions,
    requestOptions?: Partial<ProjectChatRequest>
  ): Promise<string | null> => {
    if (sessionState.isSessionActive && sessionState.sessionId) {
      return await continueStreamingSession(message, streamingOptions, requestOptions)
    } else {
      return await startNewStreamingSession(message, streamingOptions, requestOptions)
    }
  }, [sessionState.isSessionActive, sessionState.sessionId, continueStreamingSession, startNewStreamingSession])

  return {
    // Session state
    sessionId: sessionState.sessionId,
    isSessionActive: sessionState.isSessionActive,
    sessionError: sessionState.sessionError,
    
    // Streaming session operations
    startNewStreamingSession,
    continueStreamingSession,
    sendStreamingMessage,
    ...base, // Include all base operations
    
    // Mutation states
    isStartingSession: startStreamingSessionMutation.isPending,
    isContinuingSession: continueStreamingSessionMutation.isPending,
    startSessionError: startStreamingSessionMutation.error,
    continueSessionError: continueStreamingSessionMutation.error,
    
    // Computed states
    hasActiveSession: base.hasActiveSession,
    isLoading: startStreamingSessionMutation.isPending || continueStreamingSessionMutation.isPending,
    error: sessionState.sessionError || startStreamingSessionMutation.error || continueStreamingSessionMutation.error,
  }
}

/**
 * Simple hook for managing a session ID state
 * Useful when you just need to store and manage a session ID without API calls
 */
export const useProjectChatSessionId = (initialSessionId?: string) => {
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId || null)

  const clearSessionId = useCallback(() => {
    setSessionId(null)
  }, [])

  const hasSession = !!sessionId

  return {
    sessionId,
    setSessionId,
    clearSessionId,
    hasSession,
  }
}

/**
 * Export all session management hooks
 */
export default {
  useProjectChatSession,
  useProjectChatStreamingSession,
  useProjectChatSessionId,
}