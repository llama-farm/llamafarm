import { useState, useEffect, useCallback, useRef } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ChatboxMessage, ChatSession } from '../types/chatbox'
import { chatKeys } from './useChat'
import { SessionStorage } from '../utils/storage'



/**
 * Custom hook for managing chat session persistence and restoration
 * Provides session management with localStorage persistence
 * Now supports project-aware session management
 */
export function useChatSession(initialSessionId?: string, namespace?: string, project?: string) {
  const queryClient = useQueryClient()
  const projectKey = namespace && project ? `${namespace}/${project}` : null
  
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => {
    // Try to restore from localStorage, fallback to provided ID or empty string
    // Session ID will be provided by server on first chat message
    const saved = SessionStorage.getCurrentSessionId(namespace, project)
    return saved || initialSessionId || ''
  })

  // Query for current session messages
  const { data: messages = [], isLoading, error } = useQuery({
    queryKey: chatKeys.session(currentSessionId),
    queryFn: () => loadSessionMessages(currentSessionId),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 30, // 30 minutes
    enabled: !!currentSessionId, // Don't query if no session ID
  })

  // Query for session list (project-specific)
  const { data: sessions = [] } = useQuery({
    queryKey: chatKeys.sessions(),
    queryFn: () => loadAllSessions(namespace, project),
    staleTime: 1000 * 60 * 10, // 10 minutes
  })

  // Load messages from localStorage
  const loadSessionMessages = useCallback((sessionId: string): ChatboxMessage[] => {
    return SessionStorage.getSessionMessages(sessionId)
  }, [])

  // Load all sessions from localStorage (project-specific)
  const loadAllSessions = useCallback((namespace?: string, project?: string): ChatSession[] => {
    return SessionStorage.getSessionList(namespace, project)
  }, [])

  // Save messages to localStorage
  const saveSessionMessages = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    SessionStorage.setSessionMessages(sessionId, messages)
    
    // Update session metadata
    updateSessionMetadata(sessionId, messages, namespace, project)
  }, [namespace, project])

  // Update session metadata (project-specific)
  const updateSessionMetadata = useCallback((sessionId: string, messages: ChatboxMessage[], namespace?: string, project?: string) => {
    if (typeof window === 'undefined') return
    
    try {
      const sessions = loadAllSessions(namespace, project)
      const existingIndex = sessions.findIndex(s => s.id === sessionId)
      
      const sessionData: ChatSession = {
        id: sessionId,
        createdAt: existingIndex >= 0 ? sessions[existingIndex].createdAt : new Date(),
        lastActivity: new Date(),
        messageCount: messages.length,
        title: messages.length > 0 && messages[0].content
          ? messages[0].content.length > 50
            ? messages[0].content.substring(0, 50) + '...'
            : messages[0].content
          : 'New Chat'
      }
      
      if (existingIndex >= 0) {
        sessions[existingIndex] = sessionData
      } else {
        sessions.push(sessionData)
      }
      
      // Keep only the last 10 sessions
      const sortedSessions = sessions
        .sort((a, b) => b.lastActivity.getTime() - a.lastActivity.getTime())
        .slice(0, 10)
      
      SessionStorage.setSessionList(sortedSessions, namespace, project)
      
      // Invalidate sessions query to trigger refetch
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
    } catch (error) {
      console.warn('Failed to update session metadata:', error)
    }
  }, [loadAllSessions, queryClient])

  // Create new session (will get ID from server on first message)
  const createNewSession = useCallback(() => {
    const newSessionId = '' // Empty until server provides ID
    
    setCurrentSessionId(newSessionId)
    
    // Clear current session from localStorage - will be set when server provides ID
    SessionStorage.removeCurrentSessionId(namespace, project)
    
    // Invalidate queries to refresh data
    if (currentSessionId) {
      queryClient.invalidateQueries({ queryKey: chatKeys.session(currentSessionId) })
    }
    
    return newSessionId
  }, [queryClient, namespace, project, projectKey, currentSessionId])

  // Switch to existing session
  const switchToSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
    
    // Save to localStorage
    SessionStorage.setCurrentSessionId(sessionId, namespace, project)
    
    // Invalidate queries to refresh data
    queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) })
  }, [queryClient, namespace, project])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    if (typeof window === 'undefined') return
    
    try {
      // Remove messages
      SessionStorage.removeSessionMessages(sessionId)
      
      // Update session list
      const sessions = loadAllSessions(namespace, project).filter(s => s.id !== sessionId)
      SessionStorage.setSessionList(sessions, namespace, project)
      
      // If deleting current session, create a new one
      if (sessionId === currentSessionId) {
        createNewSession()
      }
      
      // Invalidate queries
      queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) })
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
    } catch (error) {
      console.warn(`Failed to delete session ${sessionId}:`, error)
    }
  }, [currentSessionId, loadAllSessions, createNewSession, queryClient, namespace, project])

  // Clear all sessions
  const clearAllSessions = useCallback(() => {
    if (typeof window === 'undefined') return
    
    try {
      // Remove all session data using utility
      SessionStorage.clearProjectSessions(namespace, project)
      
      // Create new session (server will provide ID)
      createNewSession()
      
      // Invalidate all queries
      queryClient.invalidateQueries({ queryKey: chatKeys.all })
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
      
      return '' // Session ID will be provided by server on first message
    } catch (error) {
      console.warn('Failed to clear all sessions:', error)
    }
  }, [loadAllSessions, createNewSession, queryClient, namespace, project])

  // Set session ID when received from server
  const setSessionId = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
    
    // Save to localStorage
    SessionStorage.setCurrentSessionId(sessionId, namespace, project)
    
    // Invalidate queries to refresh data
    queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) })
  }, [queryClient, namespace, project, projectKey, currentSessionId])

  // Track project context changes and load appropriate session
  const prevProjectKeyRef = useRef<string | null>(null)
  useEffect(() => {
    if (prevProjectKeyRef.current !== null && prevProjectKeyRef.current !== projectKey) {
      
        // Load session for the new project
        if (namespace && project) {
          const newProjectSession = SessionStorage.getCurrentSessionId(namespace, project)
          
          if (newProjectSession && newProjectSession !== currentSessionId) {
            setCurrentSessionId(newProjectSession)
          } else if (!newProjectSession) {
            setCurrentSessionId('')
          }
        }
    }
    prevProjectKeyRef.current = projectKey
  }, [projectKey, currentSessionId, namespace, project])

  // Save current session ID to localStorage when it changes
  useEffect(() => {
    if (currentSessionId) {
      SessionStorage.setCurrentSessionId(currentSessionId, namespace, project)
    }
  }, [currentSessionId, namespace, project, projectKey])

  return {
    // Current session state
    currentSessionId,
    messages,
    isLoading,
    error,
    
    // Session management
    sessions,
    createNewSession,
    switchToSession,
    deleteSession,
    clearAllSessions,
    setSessionId, // New function to set session ID from server
    
    // Message persistence
    saveSessionMessages,
    
    // Computed values
    hasMessages: messages.length > 0,
    currentSession: sessions.find(s => s.id === currentSessionId),
  }
}

export default useChatSession
