/**
 * Project Chat Session Management Hooks
 *
 * Provides session state management for project chat.
 * Actual chat API calls should use useChatCompletions hooks.
 */

import { useState, useCallback, useEffect } from 'react'

/**
 * Project chat session state interface
 */
export interface ProjectChatSessionState {
  sessionId: string | null
  isSessionActive: boolean
  sessionError: Error | null
}

/**
 * Simple hook for managing a streaming session ID state
 * Used by TestChat to track session IDs returned from the unified chat completions API
 */
export const useProjectChatStreamingSession = () => {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isSessionActive, setIsSessionActive] = useState(false)
  const [sessionError, setSessionError] = useState<Error | null>(null)

  // Sync isSessionActive with sessionId changes
  useEffect(() => {
    setIsSessionActive(!!sessionId)
  }, [sessionId])

  const clearSession = useCallback(() => {
    setSessionId(null)
    setIsSessionActive(false)
    setSessionError(null)
  }, [])

  const resetSessionState = useCallback(() => {
    clearSession()
  }, [clearSession])

  return {
    sessionId,
    isSessionActive,
    sessionError,
    setSessionId,
    clearSession,
    resetSessionState,
    hasActiveSession: isSessionActive && !!sessionId,
  }
}

/**
 * Simple hook for managing a session ID state
 * Useful when you just need to store and manage a session ID without API calls
 */
export const useProjectChatSessionId = (initialSessionId?: string) => {
  const [sessionId, setSessionId] = useState<string | null>(
    initialSessionId || null
  )

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
  useProjectChatStreamingSession,
  useProjectChatSessionId,
}
