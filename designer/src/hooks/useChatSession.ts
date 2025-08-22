import { useState, useEffect, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ChatboxMessage, ChatSession } from '../types/chatbox'
import { generateSessionId } from '../utils/idGenerator'
import { chatKeys } from './useChat'

// Session storage keys
const SESSION_STORAGE_KEYS = {
  CURRENT_SESSION: 'chatbox_current_session',
  SESSION_MESSAGES: (sessionId: string) => `chatbox_messages_${sessionId}`,
  SESSION_LIST: 'chatbox_sessions',
} as const



/**
 * Custom hook for managing chat session persistence and restoration
 * Provides session management with localStorage persistence
 */
export function useChatSession(initialSessionId?: string) {
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string>(() => {
    // Try to restore from localStorage, fallback to provided or generated ID
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(SESSION_STORAGE_KEYS.CURRENT_SESSION)
      return saved || initialSessionId || generateSessionId()
    }
    return initialSessionId || generateSessionId()
  })

  // Query for current session messages
  const { data: messages = [], isLoading, error } = useQuery({
    queryKey: chatKeys.session(currentSessionId),
    queryFn: () => loadSessionMessages(currentSessionId),
    staleTime: 1000 * 60 * 5, // 5 minutes
    gcTime: 1000 * 60 * 30, // 30 minutes
  })

  // Query for session list
  const { data: sessions = [] } = useQuery({
    queryKey: chatKeys.sessions(),
    queryFn: () => loadAllSessions(),
    staleTime: 1000 * 60 * 10, // 10 minutes
  })

  // Load messages from localStorage
  const loadSessionMessages = useCallback((sessionId: string): ChatboxMessage[] => {
    if (typeof window === 'undefined') return []
    
    try {
      const stored = localStorage.getItem(SESSION_STORAGE_KEYS.SESSION_MESSAGES(sessionId))
      if (!stored) return []
      
      const parsed = JSON.parse(stored)
      // Convert timestamp strings back to Date objects
      return parsed.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }))
    } catch (error) {
      console.warn(`Failed to load messages for session ${sessionId}:`, error)
      return []
    }
  }, [])

  // Load all sessions from localStorage
  const loadAllSessions = useCallback((): ChatSession[] => {
    if (typeof window === 'undefined') return []
    
    try {
      const stored = localStorage.getItem(SESSION_STORAGE_KEYS.SESSION_LIST)
      if (!stored) return []
      
      const parsed = JSON.parse(stored)
      return parsed.map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastActivity: new Date(session.lastActivity)
      }))
    } catch (error) {
      console.warn('Failed to load session list:', error)
      return []
    }
  }, [])

  // Save messages to localStorage
  const saveSessionMessages = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    if (typeof window === 'undefined') return
    
    try {
      localStorage.setItem(
        SESSION_STORAGE_KEYS.SESSION_MESSAGES(sessionId),
        JSON.stringify(messages)
      )
      
      // Update session metadata
      updateSessionMetadata(sessionId, messages)
    } catch (error) {
      console.warn(`Failed to save messages for session ${sessionId}:`, error)
    }
  }, [])

  // Update session metadata
  const updateSessionMetadata = useCallback((sessionId: string, messages: ChatboxMessage[]) => {
    if (typeof window === 'undefined') return
    
    try {
      const sessions = loadAllSessions()
      const existingIndex = sessions.findIndex(s => s.id === sessionId)
      
      const sessionData: ChatSession = {
        id: sessionId,
        createdAt: existingIndex >= 0 ? sessions[existingIndex].createdAt : new Date(),
        lastActivity: new Date(),
        messageCount: messages.length,
        title: messages.length > 0 && messages[0].content ? messages[0].content.substring(0, 50) + '...' : 'New Chat'
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
      
      localStorage.setItem(SESSION_STORAGE_KEYS.SESSION_LIST, JSON.stringify(sortedSessions))
      
      // Invalidate sessions query to trigger refetch
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
    } catch (error) {
      console.warn('Failed to update session metadata:', error)
    }
  }, [loadAllSessions, queryClient])

  // Create new session
  const createNewSession = useCallback(() => {
    const newSessionId = generateSessionId()
    setCurrentSessionId(newSessionId)
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEYS.CURRENT_SESSION, newSessionId)
    }
    
    // Invalidate queries to refresh data
    queryClient.invalidateQueries({ queryKey: chatKeys.session(newSessionId) })
    
    return newSessionId
  }, [queryClient])

  // Switch to existing session
  const switchToSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
    
    // Save to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEYS.CURRENT_SESSION, sessionId)
    }
    
    // Invalidate queries to refresh data
    queryClient.invalidateQueries({ queryKey: chatKeys.session(sessionId) })
  }, [queryClient])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    if (typeof window === 'undefined') return
    
    try {
      // Remove messages
      localStorage.removeItem(SESSION_STORAGE_KEYS.SESSION_MESSAGES(sessionId))
      
      // Update session list
      const sessions = loadAllSessions().filter(s => s.id !== sessionId)
      localStorage.setItem(SESSION_STORAGE_KEYS.SESSION_LIST, JSON.stringify(sessions))
      
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
  }, [currentSessionId, loadAllSessions, createNewSession, queryClient])

  // Clear all sessions
  const clearAllSessions = useCallback(() => {
    if (typeof window === 'undefined') return
    
    try {
      // Remove all session data
      const sessions = loadAllSessions()
      sessions.forEach(session => {
        localStorage.removeItem(SESSION_STORAGE_KEYS.SESSION_MESSAGES(session.id))
      })
      
      localStorage.removeItem(SESSION_STORAGE_KEYS.SESSION_LIST)
      localStorage.removeItem(SESSION_STORAGE_KEYS.CURRENT_SESSION)
      
      // Create new session
      const newSessionId = createNewSession()
      
      // Invalidate all queries
      queryClient.invalidateQueries({ queryKey: chatKeys.all })
      queryClient.invalidateQueries({ queryKey: chatKeys.sessions() })
      
      return newSessionId
    } catch (error) {
      console.warn('Failed to clear all sessions:', error)
    }
  }, [loadAllSessions, createNewSession, queryClient])

  // Save current session ID to localStorage when it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(SESSION_STORAGE_KEYS.CURRENT_SESSION, currentSessionId)
    }
  }, [currentSessionId])

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
    
    // Message persistence
    saveSessionMessages,
    
    // Computed values
    hasMessages: messages.length > 0,
    currentSession: sessions.find(s => s.id === currentSessionId),
  }
}

export default useChatSession
