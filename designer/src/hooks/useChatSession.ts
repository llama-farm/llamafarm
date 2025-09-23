import { useState, useEffect, useCallback } from 'react'
import { useActiveProject } from './useActiveProject'
import {
  findExistingSession,
} from '../utils/projectSessionManager'
import { ChatboxMessage } from '../types/chatbox'

/**
 * Hook that provides session management for chatbox with the expected API
 * Phase 1: Simplified stub version - functionality temporarily disabled
 */
export function useChatSession(initialSessionId?: string) {
  const activeProject = useActiveProject()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(initialSessionId || null)
  const [persistedMessages, setPersistedMessages] = useState<ChatboxMessage[]>([])
  const [isLoading] = useState(false)

  // Phase 1 stub: Load session when activeProject changes
  useEffect(() => {
    if (!activeProject) {
      setCurrentSessionId(null)
      setPersistedMessages([])
      return
    }

    // Phase 1: Check for existing session using simplified function
    const existingSessionId = findExistingSession(
      activeProject.namespace,
      activeProject.project,
      'designer'
    )

    if (existingSessionId) {
      setCurrentSessionId(existingSessionId)
      // Phase 1: No message loading yet - will be implemented in Phase 2
      setPersistedMessages([])
    } else if (initialSessionId) {
      setCurrentSessionId(initialSessionId)
      setPersistedMessages([])
    } else {
      setCurrentSessionId(null)
      setPersistedMessages([])
    }
  }, [activeProject?.namespace, activeProject?.project, initialSessionId])

  // Phase 1 stub: Save messages (no-op for now)
  const saveSessionMessages = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    console.log('Phase 1 stub: saveSessionMessages called', { sessionId, messageCount: messages.length })
    // Phase 1: No saving yet - will be implemented in Phase 2
  }, [])

  // Phase 1 stub: Create new session (returns null for now)
  const createNewSession = useCallback(() => {
    if (!activeProject) {
      console.warn('Cannot create session without active project')
      return null
    }

    console.log('Phase 1 stub: createNewSession called')
    // Phase 1: No session creation yet - will be implemented in Phase 2
    return null
  }, [activeProject])

  // Phase 1 stub: Set session ID (basic implementation)
  const setSessionId = useCallback((sessionId: string) => {
    console.log('Phase 1 stub: setSessionId called', sessionId)
    setCurrentSessionId(sessionId)
  }, [])

  return {
    currentSessionId,
    messages: persistedMessages,
    saveSessionMessages,
    createNewSession,
    setSessionId,
    isLoading,
  }
}
