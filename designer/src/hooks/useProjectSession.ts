/**
 * React hook for project session management
 * 
 * Manages session state for both Designer Chat and Project Chat services
 * with localStorage persistence and project context switching
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useActiveProject } from './useActiveProject'
import {
  getExistingSession,
  createSessionFromServer,
  loadChatHistory,
  addMessageToHistory,
  clearChatHistory,
  deleteSession,
  getSessionsForContext,
  type ChatMessage,
  type SessionMetadata,
} from '../utils/projectSessionManager'

export interface ProjectSessionOptions {
  chatService: 'designer' | 'project'
  autoCreate?: boolean
}

export interface ProjectSessionState {
  sessionId: string | null
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
}

export interface ProjectSessionActions {
  addMessage: (content: string, role: 'user' | 'assistant') => ChatMessage
  clearHistory: () => void
  deleteCurrentSession: () => void
  refreshSession: () => void
  createSessionFromServer: (serverSessionId: string) => void
}

/**
 * Hook for managing project sessions with project context integration
 */
export function useProjectSession(
  options: ProjectSessionOptions
): ProjectSessionState & ProjectSessionActions {
  const { chatService, autoCreate = true } = options
  const activeProject = useActiveProject()
  
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Memoize project key to avoid unnecessary effects
  const projectKey = useMemo(() => {
    if (!activeProject) return null
    return `${activeProject.namespace}/${activeProject.project}/${chatService}`
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  // Load existing session when project context changes
  useEffect(() => {
    if (!activeProject || !projectKey) {
      setSessionId(null)
      setMessages([])
      return
    }
    
    setIsLoading(true)
    setError(null)
    
    try {
      // Look for existing session (no auto-creation)
      const existingSessionId = getExistingSession(
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      
      if (existingSessionId) {
        setSessionId(existingSessionId)
        const history = loadChatHistory(existingSessionId)
        setMessages(history)
      } else {
        // No existing session - will be created on first message
        setSessionId(null)
        setMessages([])
      }
    } catch (err) {
      console.error('Failed to load session:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to load session'
      setError(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }, [projectKey])
  
  // Add message to history
  const addMessage = useCallback((content: string, role: 'user' | 'assistant'): ChatMessage => {
    if (!sessionId) {
      throw new Error('No active session')
    }
    
    const message = addMessageToHistory(sessionId, {
      role,
      content,
      timestamp: new Date().toISOString(),
    })
    
    setMessages(prev => [...prev, message])
    return message
  }, [sessionId])
  
  // Clear chat history
  const clearHistory = useCallback(() => {
    if (!sessionId) return
    
    try {
      clearChatHistory(sessionId)
      setMessages([])
    } catch (err) {
      console.error('Failed to clear history:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to clear history'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId])
  
  // Delete current session
  const deleteCurrentSession = useCallback(() => {
    if (!sessionId) return
    
    try {
      deleteSession(sessionId)
      setSessionId(null)
      setMessages([])
      
      // Note: New sessions will be created on next message send
    } catch (err) {
      console.error('Failed to delete session:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete session'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId, autoCreate, projectKey])
  
  // Refresh session (reload from storage)
  const refreshSession = useCallback(() => {
    if (!sessionId) return
    
    try {
      const history = loadChatHistory(sessionId)
      setMessages(history)
      setError(null)
    } catch (err) {
      console.error('Failed to refresh session:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to refresh session'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [sessionId])
  
  // Create session with server-provided session ID
  const createSessionFromServerCallback = useCallback((serverSessionId: string) => {
    if (!activeProject) {
      throw new Error('No active project')
    }
    
    try {
      createSessionFromServer(
        serverSessionId,
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      
      setSessionId(serverSessionId)
      setMessages([])
      setError(null)
    } catch (err) {
      console.error('Failed to create session from server:', err)
      const errorMessage = err instanceof Error ? err.message : 'Failed to create session from server'
      setError(errorMessage)
      throw new Error(errorMessage)
    }
  }, [projectKey])
  
  return {
    // State
    sessionId,
    messages,
    isLoading,
    error,
    
    // Actions
    addMessage,
    clearHistory,
    deleteCurrentSession,
    refreshSession,
    createSessionFromServer: createSessionFromServerCallback,
  }
}

/**
 * Hook for getting sessions for the current project context
 */
export function useProjectSessions(chatService?: 'designer' | 'project') {
  const activeProject = useActiveProject()
  const [sessions, setSessions] = useState<Array<{sessionId: string, metadata: SessionMetadata}>>([])
  const [isLoading, setIsLoading] = useState(false)
  
  const refreshSessions = useCallback(() => {
    if (!activeProject) {
      setSessions([])
      return
    }
    
    setIsLoading(true)
    try {
      const projectSessions = getSessionsForContext(
        activeProject.namespace,
        activeProject.project,
        chatService
      )
      setSessions(projectSessions)
    } catch (err) {
      console.error('Failed to load sessions:', err)
      throw new Error(err instanceof Error ? err.message : 'Failed to load sessions')
    } finally {
      setIsLoading(false)
    }
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  useEffect(() => {
    refreshSessions()
  }, [activeProject?.namespace, activeProject?.project, chatService])
  
  return {
    sessions,
    isLoading,
    refreshSessions,
  }
}

export default useProjectSession